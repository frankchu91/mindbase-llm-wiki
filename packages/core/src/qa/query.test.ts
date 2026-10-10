import { describe, it, expect, beforeEach } from 'vitest';
import { MemoryStore } from '../storage/memory_store';
import { SearchIndex } from '../search/index';
import { askQuestion, preferSources, loadCandidateSummaries, type QAEvent, type CitedSource } from './query';
import type { SearchResult } from '../search/index';
import type { ChatChunk, ChatRequest, MetaJson } from '../types';
import type { LLMAdapter } from '../adapters/types';

function fakeAdapter(script: ChatChunk[][]): LLMAdapter & { calls: ChatRequest[] } {
  let i = 0;
  const calls: ChatRequest[] = [];
  return {
    name: 'openai',
    supportsTools: true,
    calls,
    async *chat(req: ChatRequest) {
      calls.push(req);
      const batch = script[i] ?? [];
      i += 1;
      for (const c of batch) yield c;
    },
    estimateTokens: (t: string) => t.length,
    async testConnection() {
      return { ok: true };
    },
  };
}

async function seedWiki(store: MemoryStore): Promise<SearchIndex> {
  await store.writeText('wiki/INDEX.md', '# MindBase Wiki Index\n\n- [RAG](wiki/notes/rag.md) — overview\n');
  await store.writeText('wiki/notes/rag.md', '# RAG\n\nRetrieval-augmented generation uses external retrieval.');
  const meta: MetaJson = {
    id: 'concept-rag',
    type: 'concept',
    title: 'RAG',
    created: '2026-04-08T00:00:00Z',
    updated: '2026-04-08T00:00:00Z',
    sources: ['r1'],
    related: [],
    one_liner: 'Retrieval augmented generation',
    word_count: 10,
    compile_version: 1,
    edit_state: 'auto',
    last_human_edit: null,
  };
  await store.writeJSON('wiki/notes/rag.meta.json', meta);
  const idx = new SearchIndex();
  idx.add({ path: 'wiki/notes/rag.md', title: 'RAG', body: 'retrieval generation', type: 'concept' });
  return idx;
}

describe('askQuestion', () => {
  let store: MemoryStore;
  let idx: SearchIndex;

  beforeEach(async () => {
    store = new MemoryStore();
    idx = await seedWiki(store);
  });

  it('yields progress events then answer deltas then a done event with citations', async () => {
    const adapter = fakeAdapter([
      [
        {
          kind: 'tool_call',
          tool_call: {
            id: 'c1',
            name: 'read_file',
            arguments: { path: 'wiki/notes/rag.md' },
          },
        },
        { kind: 'done', usage: { input_tokens: 20, output_tokens: 5 } },
      ],
      [
        { kind: 'delta', text: 'RAG combines retrieval with generation [1].' },
        { kind: 'done', usage: { input_tokens: 50, output_tokens: 10 } },
      ],
    ]);

    const events: QAEvent[] = [];
    for await (const e of askQuestion({
      question: 'What is RAG?',
      store,
      index: idx,
      adapter,
      model: 'gpt-4o-mini',
    })) {
      events.push(e);
    }

    const phases = events.filter((e) => e.kind === 'progress').map((e) => (e as { kind: 'progress'; phase: string }).phase);
    expect(phases).toContain('read_index');
    expect(phases).toContain('keyword_filter');
    expect(phases).toContain('llm_call');

    const deltas = events.filter((e) => e.kind === 'delta').map((e) => (e as { kind: 'delta'; text: string }).text).join('');
    expect(deltas).toContain('RAG combines retrieval');

    const done = events.find((e) => e.kind === 'done') as { kind: 'done'; citations: Array<{ path: string }>; usage: { input_tokens: number; output_tokens: number } } | undefined;
    expect(done).toBeDefined();
    expect(done?.citations?.length ?? 0).toBeGreaterThan(0);
    expect(done?.usage.output_tokens).toBeGreaterThan(0);
  });

  it('handles LLM returning no tool calls (answer directly from context)', async () => {
    const adapter = fakeAdapter([
      [
        { kind: 'delta', text: 'RAG is retrieval-augmented generation.' },
        { kind: 'done', usage: { input_tokens: 10, output_tokens: 5 } },
      ],
    ]);
    const events: QAEvent[] = [];
    for await (const e of askQuestion({
      question: 'What is RAG?',
      store,
      index: idx,
      adapter,
      model: 'gpt-4o-mini',
    })) {
      events.push(e);
    }
    const text = events.filter((e) => e.kind === 'delta').map((e) => (e as { kind: 'delta'; text: string }).text).join('');
    expect(text).toContain('retrieval-augmented');
  });

  it('emits sources event exactly once, before any delta, with stable 1-indexed entries', async () => {
    const adapter = fakeAdapter([
      [
        { kind: 'delta', text: 'RAG is retrieval-augmented generation [1].' },
        { kind: 'done', usage: { input_tokens: 10, output_tokens: 5 } },
      ],
    ]);
    const events: QAEvent[] = [];
    for await (const e of askQuestion({
      question: 'What is RAG?',
      store,
      index: idx,
      adapter,
      model: 'gpt-4o-mini',
    })) {
      events.push(e);
    }

    const sourceEvents = events.filter((e) => e.kind === 'sources');
    expect(sourceEvents.length).toBe(1);

    const sourcesEvent = sourceEvents[0] as { kind: 'sources'; sources: CitedSource[] };
    expect(sourcesEvent.sources.length).toBeGreaterThan(0);

    // Verify 1-indexed and stable n values
    sourcesEvent.sources.forEach((s, i) => {
      expect(s.n).toBe(i + 1);
      expect(s.slug).toBeTruthy();
      expect(s.title).toBeTruthy();
      expect(s.path).toBeTruthy();
    });

    // Verify sources event comes BEFORE any delta event
    const sourcesIdx = events.indexOf(sourceEvents[0]!);
    const firstDeltaIdx = events.findIndex((e) => e.kind === 'delta');
    expect(sourcesIdx).toBeLessThan(firstDeltaIdx);
  });

  it('includes sources in the done event matching the sources event', async () => {
    const adapter = fakeAdapter([
      [
        { kind: 'delta', text: 'RAG is retrieval-augmented generation [1].' },
        { kind: 'done', usage: { input_tokens: 10, output_tokens: 5 } },
      ],
    ]);
    const events: QAEvent[] = [];
    for await (const e of askQuestion({
      question: 'What is RAG?',
      store,
      index: idx,
      adapter,
      model: 'gpt-4o-mini',
    })) {
      events.push(e);
    }

    const sourcesEvent = events.find((e) => e.kind === 'sources') as { kind: 'sources'; sources: CitedSource[] } | undefined;
    const doneEvent = events.find((e) => e.kind === 'done') as { kind: 'done'; citations: unknown[]; sources: CitedSource[]; usage: { input_tokens: number; output_tokens: number } } | undefined;

    expect(sourcesEvent).toBeDefined();
    expect(doneEvent).toBeDefined();
    expect(doneEvent?.sources).toBeDefined();
    expect(doneEvent?.sources.length).toBe(sourcesEvent?.sources.length);

    // Sources in done event should match those in the sources event
    doneEvent?.sources.forEach((s, i) => {
      expect(s.n).toBe(sourcesEvent!.sources[i]!.n);
      expect(s.slug).toBe(sourcesEvent!.sources[i]!.slug);
      expect(s.title).toBe(sourcesEvent!.sources[i]!.title);
    });
  });

  it('propagates error chunks as error events', async () => {
    const adapter = fakeAdapter([[{ kind: 'error', error: 'rate limit' }]]);
    const events: QAEvent[] = [];
    for await (const e of askQuestion({
      question: 'What is RAG?',
      store,
      index: idx,
      adapter,
      model: 'gpt-4o-mini',
    })) {
      events.push(e);
    }
    const err = events.find((e) => e.kind === 'error') as { kind: 'error'; error: string } | undefined;
    expect(err?.error).toMatch(/rate limit/);
  });

  it('tags v2 source-layer hits as [source] and wiki pages as [wiki]', async () => {
    await store.writeText(
      'sources/contributors/alice/2026-09-01.md',
      '# Alice daily 2026-09-01\n\nI decided RAG is not enough; we need a maintained wiki.\n',
    );
    await store.writeText(
      'sources/research/rag-vs-wiki.md',
      '# RAG vs Wiki\n\nAI synthesis comparing retrieval and compiled wikis.\n',
    );
    idx.add({ path: 'sources/contributors/alice/2026-09-01.md', title: 'Alice daily 2026-09-01', body: 'RAG retrieval wiki decided', type: 'source' });
    idx.add({ path: 'sources/research/rag-vs-wiki.md', title: 'RAG vs Wiki', body: 'RAG retrieval wiki synthesis', type: 'concept' });

    const adapter = fakeAdapter([
      [
        { kind: 'delta', text: 'RAG is retrieval [1].' },
        { kind: 'done', usage: { input_tokens: 10, output_tokens: 5 } },
      ],
    ]);
    const events: QAEvent[] = [];
    for await (const e of askQuestion({
      question: 'RAG retrieval wiki',
      store,
      index: idx,
      adapter,
      model: 'gpt-4o-mini',
    })) {
      events.push(e);
    }

    const sourcesEvent = events.find((e) => e.kind === 'sources') as { kind: 'sources'; sources: CitedSource[] };
    const contributor = sourcesEvent.sources.find((s) => s.path === 'sources/contributors/alice/2026-09-01.md');
    const research = sourcesEvent.sources.find((s) => s.path === 'sources/research/rag-vs-wiki.md');
    const legacy = sourcesEvent.sources.find((s) => s.path === 'wiki/notes/rag.md');
    expect(contributor?.layer).toBe('source');
    expect(contributor?.title).toBe('Alice daily 2026-09-01');
    expect(contributor?.one_liner).toBe('I decided RAG is not enough; we need a maintained wiki.');
    expect(research?.layer).toBe('wiki');
    expect(legacy?.layer).toBe('wiki');

    // The prompt must carry the layer tag per candidate + the LAYERS instruction.
    const prompt = String(adapter.calls[0]?.messages[0]?.content ?? '');
    expect(prompt).toMatch(/\[\d+\] \[source\] Alice daily 2026-09-01 \(sources\/contributors\/alice\/2026-09-01\.md\)/);
    expect(prompt).toMatch(/\[\d+\] \[wiki\] RAG vs Wiki \(sources\/research\/rag-vs-wiki\.md\)/);
    expect(prompt).toContain('--- [source] Alice daily 2026-09-01 ---');
    expect(prompt).toContain('LAYERS:');
  });
});

describe('askQuestion budget', () => {
  type DoneEvent = Extract<QAEvent, { kind: 'done' }>;

  async function runAsk(store: MemoryStore, idx: SearchIndex, maxSourceChars?: number): Promise<DoneEvent> {
    const adapter = fakeAdapter([
      [
        { kind: 'delta', text: 'Answer [1].' },
        { kind: 'done', usage: { input_tokens: 10, output_tokens: 5 } },
      ],
    ]);
    const events: QAEvent[] = [];
    for await (const e of askQuestion({
      question: 'What is RAG?',
      store,
      index: idx,
      adapter,
      model: 'gpt-4o-mini',
      maxSourceChars,
    })) {
      events.push(e);
    }
    const done = events.find((e): e is DoneEvent => e.kind === 'done');
    if (!done) throw new Error('no done event');
    return done;
  }

  // Tier 3 resolves raw sources via conceptMetaPath (wiki/concepts/), while
  // seedWiki writes the meta under wiki/notes/ — mirror it so tier 3 fires.
  async function seedRaw(store: MemoryStore, rawBody: string): Promise<void> {
    const meta = await store.readJSON<MetaJson>('wiki/notes/rag.meta.json');
    await store.writeJSON('wiki/concepts/rag.meta.json', meta);
    await store.writeText('raw/2026-04-08/r1.md', rawBody);
  }

  it('reports clipped paths and sent < retrieved when a long body exceeds a small budget', async () => {
    const store = new MemoryStore();
    const idx = await seedWiki(store);
    const longBody = `# RAG\n\n${'retrieval '.repeat(100)}`; // ~1000 chars
    await store.writeText('wiki/notes/rag.md', longBody);

    const done = await runAsk(store, idx, 200);

    expect(done.budget.retrievedChars).toBe(longBody.length);
    expect(done.budget.sentChars).toBe(200);
    expect(done.budget.sentChars).toBeLessThan(done.budget.retrievedChars);
    expect(done.budget.clipped).toEqual(['wiki/notes/rag.md']);
  });

  it('reports nothing clipped and sent === retrieved under a comfortable budget', async () => {
    const store = new MemoryStore();
    const idx = await seedWiki(store);
    const rawBody = '# Raw r1\n\nShort raw source body.';
    await seedRaw(store, rawBody);
    const noteBody = await store.readText('wiki/notes/rag.md');

    const done = await runAsk(store, idx, 50000);

    expect(done.budget.retrievedChars).toBe(noteBody.length + rawBody.length);
    expect(done.budget.sentChars).toBe(done.budget.retrievedChars);
    expect(done.budget.clipped).toEqual([]);
  });

  it('marks a raw source as clipped when it is dropped entirely for lack of budget', async () => {
    const store = new MemoryStore();
    const idx = await seedWiki(store);
    const rawBody = `# Raw r1\n\n${'x'.repeat(500)}`;
    await seedRaw(store, rawBody);
    const noteBody = await store.readText('wiki/notes/rag.md');

    // Note fits (~60 chars) but the tier-3 raw needs usedChars + 2000 < budget.
    const done = await runAsk(store, idx, 1000);

    expect(done.budget.retrievedChars).toBe(noteBody.length + rawBody.length);
    expect(done.budget.sentChars).toBe(noteBody.length);
    expect(done.budget.clipped).toEqual(['raw:r1']);
  });
});

describe('preferSources', () => {
  const research: SearchResult = { path: 'sources/research/a.md', title: 'a', type: 'concept', score: 10 };
  const contributor: SearchResult = { path: 'sources/contributors/u/n.md', title: 'n', type: 'source', score: 9 };

  it('boosts source-layer hits by 1.2x so a close source outranks a wiki page', () => {
    const out = preferSources([research, contributor], 2);
    expect(out.map((h) => h.path)).toEqual(['sources/contributors/u/n.md', 'sources/research/a.md']);
    expect(out[0]?.score).toBeCloseTo(10.8);
    expect(out[1]?.score).toBe(10);
  });

  it('keeps a weak source behind a strong wiki page', () => {
    const weak: SearchResult = { ...contributor, score: 5 };
    const out = preferSources([research, weak], 2);
    expect(out.map((h) => h.path)).toEqual(['sources/research/a.md', 'sources/contributors/u/n.md']);
    expect(out[1]?.score).toBeCloseTo(6);
  });

  it('considers 2x limit candidates then slices to limit', () => {
    const wiki = (i: number): SearchResult => ({ path: `sources/research/w${i}.md`, title: `w${i}`, type: 'concept', score: 10 - i * 0.1 });
    // 4 wiki hits then a source at rank 5: with limit 2 the pool is the top 4, so
    // the source is never considered; with limit 3 (pool 6) it is boosted to the top.
    const hits = [wiki(0), wiki(1), wiki(2), wiki(3), { ...contributor, score: 9.5 }];
    expect(preferSources(hits, 2).map((h) => h.path)).toEqual(['sources/research/w0.md', 'sources/research/w1.md']);
    const out3 = preferSources(hits, 3);
    expect(out3).toHaveLength(3);
    expect(out3[0]?.path).toBe('sources/contributors/u/n.md');
  });

  it('does not mutate the input hits', () => {
    const input = [{ ...contributor }];
    preferSources(input, 1);
    expect(input[0]?.score).toBe(9);
  });
});

describe('loadCandidateSummaries', () => {
  it('reads title from H1 and one_liner from first body line for non-wiki paths', async () => {
    const store = new MemoryStore();
    await store.writeText(
      'sources/contributors/bob/notes/pricing.md',
      '\n# Pricing thoughts\n\n## Context\n\n   We should charge per seat, not per project.   \nMore text.\n',
    );
    await store.writeText('sources/raw/2026-09-01/abc.extracted.md', 'no heading here\nsecond line');
    const out = await loadCandidateSummaries(store, [
      'sources/contributors/bob/notes/pricing.md',
      'sources/raw/2026-09-01/abc.extracted.md',
    ]);
    expect(out).toEqual([
      {
        path: 'sources/contributors/bob/notes/pricing.md',
        title: 'Pricing thoughts',
        one_liner: 'We should charge per seat, not per project.',
      },
      { path: 'sources/raw/2026-09-01/abc.extracted.md', title: 'abc.extracted', one_liner: 'no heading here' },
    ]);
  });

  it('truncates the one_liner to 160 chars', async () => {
    const store = new MemoryStore();
    await store.writeText('sources/contributors/bob/2026-09-02.md', `# D\n\n${'x'.repeat(300)}\n`);
    const out = await loadCandidateSummaries(store, ['sources/contributors/bob/2026-09-02.md']);
    expect(out[0]?.one_liner).toHaveLength(160);
  });

  it('keeps the wiki/ meta.json behavior', async () => {
    const store = new MemoryStore();
    await store.writeText('wiki/concepts/rag.md', '# Not this title\n\nbody');
    await store.writeJSON('wiki/concepts/rag.meta.json', { title: 'RAG', one_liner: 'Retrieval augmented generation' });
    const out = await loadCandidateSummaries(store, ['wiki/concepts/rag.md']);
    expect(out).toEqual([{ path: 'wiki/concepts/rag.md', title: 'RAG', one_liner: 'Retrieval augmented generation' }]);
  });
});
