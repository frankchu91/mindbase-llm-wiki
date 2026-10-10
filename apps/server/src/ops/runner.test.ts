import { describe, it, expect, beforeEach } from 'vitest';
import { mkdtemp, mkdir, writeFile, readFile, readdir, utimes } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  runContributePlan, applyContributePlan, runBuild, runLint, runResearch,
  latestLintArtifact, dismissLintFinding, verifyEvidence,
  type OpEvent, type OpsCtx, type StoredFinding,
} from './runner';
import type { ChatChunk, ChatMessage } from '@mindbase/core';

let root: string;

/**
 * Fake LLM that replays `outputs` in order. When `captured` is given, every
 * request's messages are pushed onto it so tests can assert on the prompt.
 */
function scriptedCtx(outputs: string[], captured?: ChatMessage[][]): OpsCtx {
  let call = 0;
  return {
    projectRoot: root,
    projectId: 'test-proj',
    user: 'u',
    config: { model: 'fake-model' },
    getAdapter: () => ({
      chat: (req: { model: string; messages: ChatMessage[] }): AsyncIterable<ChatChunk> => {
        captured?.push(req.messages);
        const text = outputs[Math.min(call++, outputs.length - 1)]!;
        return (async function* () {
          yield { kind: 'delta', text } as ChatChunk;
        })();
      },
    }),
    findRelated: async () => [],
  };
}

function collect(): { events: OpEvent[]; emit: (e: OpEvent) => void } {
  const events: OpEvent[] = [];
  return { events, emit: (e) => events.push(e) };
}

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), 'mb-runner-'));
  await mkdir(join(root, 'sources', 'contributors', 'u'), { recursive: true });
  await mkdir(join(root, 'sources', 'research'), { recursive: true });
  await writeFile(join(root, 'context.md'), '# T — Context\n\n## Learnings\n\n- old\n', 'utf-8');
});

describe('contribute plan → apply', () => {
  const planJson = JSON.stringify({
    takeaways: ['t1', 't2'],
    plan: [
      { kind: 'create_research_page', slug: 'new-idea', markdown: '# New Idea\n\nbody' },
      { kind: 'append_context_section', section: 'Learnings', markdown: '- fresh learning' },
    ],
  });

  it('emits plan, applies selected actions, logs', async () => {
    const c1 = collect();
    await runContributePlan(scriptedCtx([planJson]), 'my new thought', c1.emit);
    const plan = c1.events.find((e) => e.kind === 'plan');
    expect(plan && plan.kind === 'plan' && plan.plan).toHaveLength(2);
    const planId = plan!.kind === 'plan' ? plan!.planId : '';

    const c2 = collect();
    await applyContributePlan(planId, [0, 1], c2.emit);
    const applied = c2.events.find((e) => e.kind === 'applied');
    expect(applied && applied.kind === 'applied' && applied.applied).toEqual(['sources/research/new-idea.md', 'context.md']);
    const log = await readFile(join(root, 'logs', `${new Date().toISOString().slice(0, 10)}.md`), 'utf-8');
    expect(log).toMatch(/contribute \| applied 2/);
  });

  it('deselecting an action skips it', async () => {
    const c1 = collect();
    await runContributePlan(scriptedCtx([planJson]), 'thought', c1.emit);
    const plan = c1.events.find((e) => e.kind === 'plan')!;
    const planId = plan.kind === 'plan' ? plan.planId : '';
    const c2 = collect();
    await applyContributePlan(planId, [1], c2.emit);
    const applied = c2.events.find((e) => e.kind === 'applied');
    expect(applied && applied.kind === 'applied' && applied.applied).toEqual(['context.md']);
    const pages = await readdir(join(root, 'sources', 'research'));
    expect(pages).toHaveLength(0);
  });

  it('unknown planId errors cleanly', async () => {
    const c = collect();
    await applyContributePlan('nope-id', [0], c.emit);
    expect(c.events.some((e) => e.kind === 'error')).toBe(true);
  });
});

describe('contribute: source layer + duplicate guard', () => {
  const today = () => new Date().toISOString().slice(0, 10);
  const dailyRel = () => `sources/contributors/u/${today()}.md`;
  const planWith = (...slugs: string[]) => JSON.stringify({
    takeaways: ['t'],
    plan: [
      ...slugs.map((slug) => ({ kind: 'create_research_page', slug, markdown: `# ${slug}\n\nbody [@sources/contributors/u/x.md]` })),
      { kind: 'append_context_section', section: 'Learnings', markdown: '- l' },
    ],
  });
  const userPrompt = (captured: ChatMessage[][]) => captured[0]!.find((m) => m.role === 'user')!.content as string;
  const planOf = (events: OpEvent[]) => {
    const ev = events.find((e) => e.kind === 'plan');
    if (!ev || ev.kind !== 'plan') throw new Error('no plan event');
    return ev;
  };

  it('(a) free text lands in today\'s contributor file first and the prompt cites it', async () => {
    const captured: ChatMessage[][] = [];
    const c = collect();
    await runContributePlan(scriptedCtx([planWith('idea')], captured), 'a fresh thought', c.emit);
    expect(c.events.some((e) => e.kind === 'phase' && /saving to today/.test(e.phase))).toBe(true);
    const daily = await readFile(join(root, dailyRel()), 'utf-8');
    expect(daily).toContain(`# ${today()} — u\n`);
    expect(daily).toContain('a fresh thought');
    expect(userPrompt(captured)).toContain(`[@${dailyRel()}]`);
    const log = await readFile(join(root, 'logs', `${today()}.md`), 'utf-8');
    expect(log).toMatch(/contribute \| user=u bytes=15/);
    expect(planOf(c.events).notes).toBeUndefined();
  });

  it('(b) with sourcePath given, no daily file is written and the prompt cites the given path', async () => {
    const captured: ChatMessage[][] = [];
    const c = collect();
    const sourcePath = 'sources/contributors/u/notes/my-note.md';
    await runContributePlan(scriptedCtx([planWith('idea')], captured), 'note body', c.emit, { sourcePath });
    await expect(readFile(join(root, dailyRel()), 'utf-8')).rejects.toThrow();
    expect(c.events.some((e) => e.kind === 'phase' && /saving to today/.test(e.phase))).toBe(false);
    expect(userPrompt(captured)).toContain(`[@${sourcePath}]`);
    expect(userPrompt(captured)).not.toContain(dailyRel());
  });

  it('prompt lists existing research slugs so the model can update instead of duplicating', async () => {
    await writeFile(join(root, 'sources', 'research', 'deep-work.md'), '# Deep Work', 'utf-8');
    const captured: ChatMessage[][] = [];
    await runContributePlan(scriptedCtx([planWith()], captured), 'x', collect().emit);
    expect(userPrompt(captured)).toMatch(/EXISTING RESEARCH PAGES[^\n]*\n[^\n]*deep-work/);
  });

  it('(c) create_research_page whose normalized slug matches an existing file is removed with a note', async () => {
    await writeFile(join(root, 'sources', 'research', 'deep-work.md'), '# Deep Work', 'utf-8');
    const c = collect();
    // Model slug differs only by case/separator; the action schema itself
    // requires kebab-case so use a valid slug that still normalizes equal.
    await runContributePlan(scriptedCtx([planWith('deep--work')]), 'x', c.emit);
    const plan = planOf(c.events);
    expect(plan.plan.map((a) => a.kind)).toEqual(['append_context_section']);
    expect(plan.notes).toHaveLength(1);
    expect(plan.notes![0]).toContain('"deep--work"');
    expect(plan.notes![0]).toContain('sources/research/deep-work.md');
    expect(plan.notes![0]).toMatch(/already exists/);
  });

  it('(d) a slug reserved by a pending plan is refused until that plan is consumed', async () => {
    const cA = collect();
    await runContributePlan(scriptedCtx([planWith('foo')]), 'a', cA.emit);
    const planA = planOf(cA.events);
    expect(planA.plan).toHaveLength(2);
    expect(planA.notes).toBeUndefined();

    const cB = collect();
    await runContributePlan(scriptedCtx([planWith('foo')]), 'b', cB.emit);
    const planB = planOf(cB.events);
    expect(planB.plan.map((a) => a.kind)).toEqual(['append_context_section']);
    expect(planB.notes).toEqual(['Skipped creating "foo" — another pending plan is already creating it.']);

    // Apply A selecting ONLY the context append — foo.md is never written,
    // so the reservation is released and C may propose it again.
    const cApply = collect();
    await applyContributePlan(planA.planId, [1], cApply.emit);
    expect((await readdir(join(root, 'sources', 'research')))).toEqual([]);

    const cC = collect();
    await runContributePlan(scriptedCtx([planWith('foo')]), 'c', cC.emit);
    const planC = planOf(cC.events);
    expect(planC.plan.map((a) => a.kind)).toEqual(['create_research_page', 'append_context_section']);
    expect(planC.notes).toBeUndefined();

    // Now actually create it via C; a later plan D hits the on-disk guard.
    const cApplyC = collect();
    await applyContributePlan(planC.planId, [0], cApplyC.emit);
    expect((await readdir(join(root, 'sources', 'research')))).toEqual(['foo.md']);
    const cD = collect();
    await runContributePlan(scriptedCtx([planWith('foo')]), 'd', cD.emit);
    const planD = planOf(cD.events);
    expect(planD.plan.map((a) => a.kind)).toEqual(['append_context_section']);
    expect(planD.notes).toEqual(['Skipped creating "foo" — a page for this already exists (sources/research/foo.md). Update it instead.']);
  });

  it('a plan whose every action was removed is still emitted (empty plan + notes)', async () => {
    await writeFile(join(root, 'sources', 'research', 'foo.md'), '# Foo', 'utf-8');
    const onlyCreate = JSON.stringify({ takeaways: ['t'], plan: [{ kind: 'create_research_page', slug: 'foo', markdown: '# Foo' }] });
    const c = collect();
    await runContributePlan(scriptedCtx([onlyCreate]), 'x', c.emit);
    const plan = planOf(c.events);
    expect(plan.plan).toEqual([]);
    expect(plan.notes).toHaveLength(1);
    const cApply = collect();
    await applyContributePlan(plan.planId, [], cApply.emit);
    expect(cApply.events.some((e) => e.kind === 'error' && /No actions selected/.test(e.error))).toBe(true);
  });

  it('(e) pages the model forgot to cite get the source citation added deterministically', async () => {
    const src = 'sources/contributors/u/notes/thing.md';
    const uncited = JSON.stringify({ takeaways: ['t'], plan: [
      { kind: 'create_research_page', slug: 'thing', markdown: '# Thing\n\nBody.' },
      { kind: 'append_context_section', section: 'Learnings', markdown: '- learned a thing' },
      { kind: 'create_research_page', slug: 'cited', markdown: `# Cited\n\nSee [@${src}].` },
    ] });
    const c = collect();
    await runContributePlan(scriptedCtx([uncited]), 'x', c.emit, { sourcePath: src });
    const plan = planOf(c.events).plan;
    expect(plan[0]).toMatchObject({ kind: 'create_research_page', markdown: `# Thing\n\nBody.\n\nSources: [@${src}]\n` });
    expect(plan[1]).toMatchObject({ kind: 'append_context_section', markdown: `- learned a thing [@${src}]` });
    // Already-cited pages are left untouched.
    expect(plan[2]).toMatchObject({ markdown: `# Cited\n\nSee [@${src}].` });
  });
});

describe('build', () => {
  it('rewrites context with snapshot + log', async () => {
    await writeFile(join(root, 'sources', 'contributors', 'u', '2099-01-01.md'), '## 10:00\n\nnew stuff', 'utf-8');
    const out = JSON.stringify({ actions: [{ kind: 'update_context', markdown: '# T — Context\n\n## Learnings\n\n- rebuilt\n' }] });
    const c = collect();
    await runBuild(scriptedCtx([out]), c.emit);
    expect(c.events.some((e) => e.kind === 'done')).toBe(true);
    expect(await readFile(join(root, 'context.md'), 'utf-8')).toContain('rebuilt');
    expect((await readdir(join(root, 'state', 'builder', 'snapshots'))).length).toBe(1);
    const log = await readFile(join(root, 'logs', `${new Date().toISOString().slice(0, 10)}.md`), 'utf-8');
    expect(log).toMatch(/build \| applied 1/);
  });

  it('rejects a concurrent build for the same project', async () => {
    // First build's adapter never finishes until we let it; start it and
    // immediately try a second one.
    let release!: () => void;
    const gate = new Promise<void>((r) => { release = r; });
    const slowCtx: OpsCtx = {
      ...scriptedCtx(['{}']),
      getAdapter: () => ({
        chat: (): AsyncIterable<ChatChunk> =>
          (async function* () {
            await gate;
            yield { kind: 'delta', text: JSON.stringify({ actions: [{ kind: 'update_context', markdown: 'x' }] }) } as ChatChunk;
          })(),
      }),
    };
    const c1 = collect();
    const first = runBuild(slowCtx, c1.emit);
    await new Promise((r) => setTimeout(r, 20));
    const c2 = collect();
    await runBuild(scriptedCtx(['{}']), c2.emit);
    expect(c2.events.some((e) => e.kind === 'error' && /already running/.test(e.error))).toBe(true);
    release();
    await first;
  });
});

describe('research', () => {
  const researchJson = JSON.stringify({
    actions: [
      { kind: 'create_research_page', slug: 'topic-x', markdown: '# Topic X\n\nSynthesis. (source: sources/research/base.md)' },
    ],
  });

  it('wiki-only: reads related page bodies, writes the page, notes the mode', async () => {
    await writeFile(join(root, 'sources', 'research', 'base.md'), '# Base\n\nfacts', 'utf-8');
    const ctx: OpsCtx = {
      ...scriptedCtx([researchJson]),
      findRelated: async () => [{ path: 'sources/research/base.md', excerpt: 'Base' }],
    };
    const c = collect();
    await runResearch(ctx, 'topic x', c.emit);
    const applied = c.events.find((e) => e.kind === 'applied');
    expect(applied && applied.kind === 'applied' && applied.applied).toEqual(['sources/research/topic-x.md']);
    expect(applied!.kind === 'applied' && applied!.note).toMatch(/wiki only/);
    expect(await readFile(join(root, 'sources', 'research', 'topic-x.md'), 'utf-8')).toContain('Synthesis');
    const log = await readFile(join(root, 'logs', `${new Date().toISOString().slice(0, 10)}.md`), 'utf-8');
    expect(log).toMatch(/research \| "topic x" mode=wiki-only/);
  });

  it('bad brave key degrades to wiki-only instead of failing', async () => {
    const ctx: OpsCtx = { ...scriptedCtx([researchJson]), braveApiKey: 'bogus-key-for-test' };
    const c = collect();
    await runResearch(ctx, 'topic x', c.emit);
    const applied = c.events.find((e) => e.kind === 'applied');
    expect(applied && applied.kind === 'applied' && applied.applied).toEqual(['sources/research/topic-x.md']);
    expect(applied!.kind === 'applied' && applied!.note).toMatch(/Web search failed/);
  });
});

describe('lint', () => {
  const lintJson = JSON.stringify({
    findings: [
      { kind: 'orphan', pages: ['sources/research/lonely.md'], detail: 'No inbound links.' },
      { kind: 'question', pages: ['context.md'], detail: 'What is the deployment target?' },
    ],
  });

  it('emits findings, caches artifact, logs — and never writes wiki files', async () => {
    // Cited (and the cited file exists) so the deterministic unsourced_page /
    // broken_citation checks stay quiet here.
    await writeFile(join(root, 'sources', 'contributors', 'u', 'x.md'), 'source', 'utf-8');
    await writeFile(join(root, 'sources', 'research', 'lonely.md'), '# Lonely\n\nno links here [@sources/contributors/u/x.md]', 'utf-8');
    const contextBefore = await readFile(join(root, 'context.md'), 'utf-8');
    const c = collect();
    await runLint(scriptedCtx([lintJson]), c.emit);
    const ev = c.events.find((e) => e.kind === 'findings');
    expect(ev && ev.kind === 'findings' && ev.findings).toHaveLength(2);
    expect(ev!.kind === 'findings' && ev!.findings[0]!.id).toBeTruthy();
    expect(await readFile(join(root, 'context.md'), 'utf-8')).toBe(contextBefore);
    const artifact = await latestLintArtifact(root);
    expect(artifact?.findings).toHaveLength(2);
    const log = await readFile(join(root, 'logs', `${new Date().toISOString().slice(0, 10)}.md`), 'utf-8');
    expect(log).toMatch(/lint \| 2 findings/);
  });

  it('dismiss marks a finding in the cached artifact', async () => {
    await writeFile(join(root, 'sources', 'research', 'lonely.md'), '# Lonely', 'utf-8');
    const c = collect();
    await runLint(scriptedCtx([lintJson]), c.emit);
    const ev = c.events.find((e) => e.kind === 'findings')!;
    const id = ev.kind === 'findings' ? ev.findings[0]!.id : '';
    expect(await dismissLintFinding(root, id)).toBe(true);
    const artifact = await latestLintArtifact(root);
    expect(artifact?.findings[0]?.dismissed).toBe(true);
    expect(artifact?.findings[1]?.dismissed).toBe(false);
    expect(await dismissLintFinding(root, 'no-such-id')).toBe(false);
  });

  it('errors helpfully on an empty project', async () => {
    await writeFile(join(root, 'context.md'), '', 'utf-8');
    const c = collect();
    await runLint(scriptedCtx([lintJson]), c.emit);
    expect(c.events.some((e) => e.kind === 'error' && /Nothing to lint/.test(e.error))).toBe(true);
  });
});

describe('lint evidence verification', () => {
  const CITE = '[@sources/contributors/u/2026-01-01.md]';

  async function twoPages() {
    await writeFile(join(root, 'sources', 'contributors', 'u', '2026-01-01.md'), 'source', 'utf-8');
    await writeFile(join(root, 'sources', 'research', 'a.md'), `# A\n\nThe launch is scheduled for **March 2027**. ${CITE}\n`, 'utf-8');
    await writeFile(join(root, 'sources', 'research', 'b.md'), `# B\n\nThe launch is scheduled for June 2027, per the CEO. ${CITE}\n`, 'utf-8');
  }

  function findingsOf(events: OpEvent[]): StoredFinding[] {
    const ev = events.find((e) => e.kind === 'findings');
    return ev && ev.kind === 'findings' ? ev.findings : [];
  }

  it('(a) contradiction with 2 verbatim quotes is kept with verified:true', async () => {
    await twoPages();
    const out = JSON.stringify({ findings: [{
      kind: 'contradiction', pages: ['sources/research/a.md', 'b'], detail: 'Launch dates disagree.',
      evidence: [
        { page: 'sources/research/a.md', quote: 'launch is scheduled for March 2027' },
        { page: 'b', quote: 'The  launch is scheduled for *June 2027*' },
      ],
    }] });
    const c = collect();
    await runLint(scriptedCtx([out]), c.emit);
    const f = findingsOf(c.events);
    const contradiction = f.find((x) => x.kind === 'contradiction');
    expect(contradiction).toBeTruthy();
    expect(contradiction!.evidence?.map((e) => e.verified)).toEqual([true, true]);
    const artifact = await latestLintArtifact(root);
    expect(artifact?.dropped).toBe(0);
  });

  it('(b) contradiction with a fabricated quote is dropped and counted', async () => {
    await twoPages();
    const out = JSON.stringify({ findings: [
      {
        kind: 'contradiction', pages: ['sources/research/a.md', 'sources/research/b.md'], detail: 'Fabricated.',
        evidence: [
          { page: 'sources/research/a.md', quote: 'launch is scheduled for March 2027' },
          { page: 'sources/research/b.md', quote: 'the launch was cancelled entirely' },
        ],
      },
      { kind: 'orphan', pages: ['sources/research/a.md'], detail: 'No inbound links.' },
    ] });
    const c = collect();
    await runLint(scriptedCtx([out]), c.emit);
    const f = findingsOf(c.events);
    expect(f.some((x) => x.kind === 'contradiction')).toBe(false);
    expect(f.some((x) => x.kind === 'orphan')).toBe(true);
    const artifact = await latestLintArtifact(root);
    expect(artifact?.dropped).toBe(1);
    const log = await readFile(join(root, 'logs', `${new Date().toISOString().slice(0, 10)}.md`), 'utf-8');
    expect(log).toMatch(/lint \| 1 findings, pages=2 \(ui\), dropped=1/);
  });

  it('(c) stale with one real quote is kept', async () => {
    await twoPages();
    const out = JSON.stringify({ findings: [{
      kind: 'stale', pages: ['sources/research/a.md'], detail: 'Superseded by b.',
      evidence: [{ page: 'a', quote: 'scheduled for March 2027' }],
    }] });
    const c = collect();
    await runLint(scriptedCtx([out]), c.emit);
    const stale = findingsOf(c.events).find((x) => x.kind === 'stale');
    expect(stale?.evidence).toEqual([{ page: 'a', quote: 'scheduled for March 2027', verified: true }]);
  });

  it('(d) question with unverifiable evidence is kept, flagged verified:false', async () => {
    await twoPages();
    const out = JSON.stringify({ findings: [{
      kind: 'question', pages: ['CONTEXT.MD'], detail: 'Who owns the launch?',
      evidence: [{ page: './Context.md', quote: 'this text is nowhere in context' }],
    }] });
    const c = collect();
    await runLint(scriptedCtx([out]), c.emit);
    const q = findingsOf(c.events).find((x) => x.kind === 'question');
    expect(q?.pages).toEqual(['context.md']);
    expect(q?.evidence?.[0]?.verified).toBe(false);
    expect((await latestLintArtifact(root))?.dropped).toBe(0);
  });

  it('verifyEvidence resolves context.md case-insensitively and normalizes markdown', async () => {
    await writeFile(join(root, 'context.md'), '# T\n\n- We use `pnpm`   for _everything_.\n', 'utf-8');
    const base = { id: '1', dismissed: false, pages: ['context.md'], detail: 'd' };
    const { kept, dropped } = await verifyEvidence(root, [
      { ...base, kind: 'stale', evidence: [{ page: 'CONTEXT.md', quote: 'we use pnpm for everything' }] },
      { ...base, id: '2', kind: 'stale', evidence: [{ page: 'context.md', quote: 'we use yarn for everything' }] },
      { ...base, id: '3', kind: 'contradiction', evidence: [{ page: 'context.md', quote: 'we use pnpm for everything' }] },
      { ...base, id: '4', kind: 'gap' },
    ]);
    expect(kept.map((k) => k.id)).toEqual(['1', '4']);
    expect(dropped).toBe(2);
    expect(kept[0]!.evidence?.[0]?.verified).toBe(true);
  });
});

describe('lint deterministic findings', () => {
  const noFindings = JSON.stringify({ findings: [] });
  const dayAgo = Math.floor(Date.now() / 1000) - 86_400;

  it('(e) research page without citations → unsourced_page, after model findings', async () => {
    await writeFile(join(root, 'sources', 'contributors', 'u', '2026-01-01.md'), 'source', 'utf-8');
    await writeFile(join(root, 'sources', 'research', 'sourced.md'), '# S\n\nclaim [@sources/contributors/u/2026-01-01.md]\n', 'utf-8');
    await writeFile(join(root, 'sources', 'research', 'bare.md'), '# Bare\n\nno citations at all\n', 'utf-8');
    const out = JSON.stringify({ findings: [{ kind: 'orphan', pages: ['sources/research/bare.md'], detail: 'Nothing links here.' }] });
    const c = collect();
    await runLint(scriptedCtx([out]), c.emit);
    const ev = c.events.find((e) => e.kind === 'findings');
    const f = ev && ev.kind === 'findings' ? ev.findings : [];
    expect(f.map((x) => x.kind)).toEqual(['orphan', 'unsourced_page']);
    expect(f[1]!.pages).toEqual(['sources/research/bare.md']);
    expect(f[1]!.id).toBeTruthy();
    expect(f[1]!.dismissed).toBe(false);
    expect(f[1]!.evidence).toBeUndefined();
  });

  it('(f) contributor file older than context.md with no citations → uncited_source; cited one is not emitted', async () => {
    const cited = join(root, 'sources', 'contributors', 'u', '2026-01-01.md');
    const uncited = join(root, 'sources', 'contributors', 'u', '2026-01-02.md');
    const fresh = join(root, 'sources', 'contributors', 'u', '2026-01-03.md');
    await writeFile(cited, 'day one', 'utf-8');
    await writeFile(uncited, 'day two', 'utf-8');
    await writeFile(fresh, 'day three (newer than context.md — not built yet)', 'utf-8');
    await utimes(cited, dayAgo, dayAgo);
    await utimes(uncited, dayAgo, dayAgo);
    await utimes(fresh, dayAgo + 2 * 86_400, dayAgo + 2 * 86_400);
    await writeFile(join(root, 'sources', 'research', 'r.md'), '# R\n\nclaim [@sources/contributors/u/2026-01-01.md]\n', 'utf-8');
    const c = collect();
    await runLint(scriptedCtx([noFindings]), c.emit);
    const ev = c.events.find((e) => e.kind === 'findings');
    const f = ev && ev.kind === 'findings' ? ev.findings : [];
    const uncitedFindings = f.filter((x) => x.kind === 'uncited_source');
    expect(uncitedFindings.map((x) => x.pages[0])).toEqual(['sources/contributors/u/2026-01-02.md']);
    expect(f.some((x) => x.kind === 'unsourced_page')).toBe(false);
  });

  it('skips uncited_source entirely when context.md is missing', async () => {
    const { rm } = await import('node:fs/promises');
    await rm(join(root, 'context.md'));
    const old = join(root, 'sources', 'contributors', 'u', '2026-01-02.md');
    await writeFile(old, 'day two', 'utf-8');
    await utimes(old, dayAgo, dayAgo);
    await writeFile(join(root, 'sources', 'research', 'r.md'), '# R\n\n[@sources/contributors/u/zzz.md]\n', 'utf-8');
    const c = collect();
    await runLint(scriptedCtx([noFindings]), c.emit);
    const ev = c.events.find((e) => e.kind === 'findings');
    const f = ev && ev.kind === 'findings' ? ev.findings : [];
    expect(f.some((x) => x.kind === 'uncited_source')).toBe(false);
  });

  const lintFindings = async (modelOut: string = noFindings): Promise<StoredFinding[]> => {
    const c = collect();
    await runLint(scriptedCtx([modelOut]), c.emit);
    const ev = c.events.find((e) => e.kind === 'findings');
    return ev && ev.kind === 'findings' ? ev.findings : [];
  };

  it('(g) citing a file that does not exist → broken_citation naming that path; existing file → none', async () => {
    await writeFile(join(root, 'sources', 'contributors', 'u', 'here.md'), 'present', 'utf-8');
    await writeFile(join(root, 'sources', 'research', 'ok.md'), '# OK\n\n[@sources/contributors/u/here.md]\n', 'utf-8');
    await writeFile(join(root, 'sources', 'research', 'dangling.md'), '# D\n\n[@sources/contributors/u/here.md] [@sources/contributors/u/gone.md]\n', 'utf-8');
    const broken = (await lintFindings()).filter((x) => x.kind === 'broken_citation');
    expect(broken).toHaveLength(1);
    expect(broken[0]!.pages).toEqual(['sources/research/dangling.md']);
    expect(broken[0]!.detail).toBe('Cites [@sources/contributors/u/gone.md] but that file no longer exists (moved or deleted). Update the citation.');
  });

  it('(g2) context.md citing a missing file → broken_citation on context.md', async () => {
    await writeFile(join(root, 'context.md'), '# T — Context\n\n- fact [@sources/raw/2026-01-01/x.extracted.md]\n', 'utf-8');
    const broken = (await lintFindings()).filter((x) => x.kind === 'broken_citation');
    expect(broken.map((x) => x.pages)).toEqual([['context.md']]);
    expect(broken[0]!.detail).toContain('[@sources/raw/2026-01-01/x.extracted.md]');
  });

  it('(h) [[nope]] → broken_link; [[Deep_Work]] resolving to a contributor note is fine', async () => {
    await mkdir(join(root, 'sources', 'contributors', 'u', 'notes'), { recursive: true });
    await writeFile(join(root, 'sources', 'contributors', 'u', 'notes', 'deep-work.md'), '# Deep Work', 'utf-8');
    await writeFile(join(root, 'sources', 'contributors', 'u', 'src.md'), 's', 'utf-8');
    await writeFile(join(root, 'sources', 'research', 'focus.md'), '# Focus\n\nSee [[Deep_Work]] and [[nope]]. [@sources/contributors/u/src.md]\n', 'utf-8');
    const broken = (await lintFindings()).filter((x) => x.kind === 'broken_link');
    expect(broken).toHaveLength(1);
    expect(broken[0]!.pages).toEqual(['sources/research/focus.md']);
    expect(broken[0]!.detail).toBe('Links to [[nope]] but no page or note with that name exists. Create it or fix the link.');
  });

  it('(h2) a wikilink to another research page (any casing/separator) is not broken', async () => {
    await writeFile(join(root, 'sources', 'contributors', 'u', 'src.md'), 's', 'utf-8');
    await writeFile(join(root, 'sources', 'research', 'deep-work.md'), '# DW\n\n[@sources/contributors/u/src.md]\n', 'utf-8');
    await writeFile(join(root, 'sources', 'research', 'focus.md'), '# Focus\n\n[[Deep Work]] [@sources/contributors/u/src.md]\n', 'utf-8');
    expect((await lintFindings()).filter((x) => x.kind === 'broken_link')).toEqual([]);
  });

  it('(i) near-identical slugs → one near_duplicate pair; unrelated slugs → none', async () => {
    await writeFile(join(root, 'sources', 'contributors', 'u', 'src.md'), 's', 'utf-8');
    const page = (slug: string) =>
      writeFile(join(root, 'sources', 'research', `${slug}.md`), `# ${slug}\n\n[@sources/contributors/u/src.md]\n`, 'utf-8');
    await Promise.all(['attention-residue-cost', 'attention-residue-costs-explained', 'deep-work', 'attention-span'].map(page));
    const dups = (await lintFindings()).filter((x) => x.kind === 'near_duplicate');
    expect(dups).toHaveLength(1);
    expect([...dups[0]!.pages].sort()).toEqual([
      'sources/research/attention-residue-cost.md',
      'sources/research/attention-residue-costs-explained.md',
    ]);
    expect(dups[0]!.detail).toMatch(/^"attention-residue-costs?(-explained)?" and "attention-residue-costs?(-explained)?" look like the same concept\. Extend one page instead of keeping both\.$/);
  });

  it('deterministic kinds are emitted after model kinds, in the documented order', async () => {
    await mkdir(join(root, 'sources', 'contributors', 'u', 'notes'), { recursive: true });
    await writeFile(join(root, 'sources', 'research', 'alpha-beta.md'), '# AB\n\n[[nowhere]] [@sources/contributors/u/missing.md]\n', 'utf-8');
    await writeFile(join(root, 'sources', 'research', 'alpha-betas.md'), '# AB2\n\nno cites\n', 'utf-8');
    const out = JSON.stringify({ findings: [{ kind: 'question', pages: ['context.md'], detail: 'Why?' }] });
    const kinds = (await lintFindings(out)).map((x) => x.kind);
    expect(kinds).toEqual(['question', 'unsourced_page', 'broken_citation', 'broken_link', 'near_duplicate']);
  });

  it('a model-emitted broken_link (or other deterministic kind) is dropped by runLint', async () => {
    await writeFile(join(root, 'sources', 'contributors', 'u', 'src.md'), 's', 'utf-8');
    await writeFile(join(root, 'sources', 'research', 'clean.md'), '# Clean\n\n[@sources/contributors/u/src.md]\n', 'utf-8');
    const out = JSON.stringify({ findings: [
      { kind: 'broken_link', pages: ['sources/research/clean.md'], detail: 'Model thinks a link is broken.' },
      { kind: 'near_duplicate', pages: ['sources/research/clean.md'], detail: 'Model thinks this is a dup.' },
      { kind: 'broken_citation', pages: ['sources/research/clean.md'], detail: 'Model thinks a citation is broken.' },
      { kind: 'gap', pages: ['sources/research/clean.md'], detail: 'A real gap.' },
    ] });
    const f = await lintFindings(out);
    expect(f.map((x) => x.kind)).toEqual(['gap']);
  });
});

describe('contribute: fail-open when the plan fails after the entry was saved', () => {
  const throwingCtx = (): OpsCtx => ({
    ...scriptedCtx(['{}']),
    getAdapter: () => ({
      chat: (): AsyncIterable<ChatChunk> => {
        throw new Error('model unreachable');
      },
    }),
  });

  it('free text: error says the entry is saved in the daily file', async () => {
    const c = collect();
    await runContributePlan(throwingCtx(), 'a thought worth keeping', c.emit);
    const err = c.events.find((e) => e.kind === 'error');
    expect(err && err.kind === 'error' ? err.error : '').toMatch(/^Your entry is saved in sources\/contributors\/u\/\d{4}-\d{2}-\d{2}\.md — the wiki update failed: model unreachable$/);
    const daily = await readFile(join(root, 'sources', 'contributors', 'u', `${new Date().toISOString().slice(0, 10)}.md`), 'utf-8');
    expect(daily).toContain('a thought worth keeping');
  });

  it('with a caller-supplied sourcePath the plain error is kept', async () => {
    const c = collect();
    await runContributePlan(throwingCtx(), 'note body', c.emit, { sourcePath: 'sources/contributors/u/notes/n.md' });
    const err = c.events.find((e) => e.kind === 'error');
    expect(err && err.kind === 'error' ? err.error : '').toBe('model unreachable');
  });
});
