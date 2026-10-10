import { describe, it, expect, beforeEach } from 'vitest';
import { mkdtemp, mkdir, writeFile, utimes } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { z } from 'zod';
import {
  findMissingFiles, gatherProjectCore, gatherUnbuiltSources, gatherResearchPages, gatherSourceStats,
  listNoteSlugs, listResearchSlugs, parseCitations,
} from './gather';
import { completeJson } from './llm';
import type { ChatChunk, ChatMessage } from '@mindbase/core';

let root: string;

async function touch(rel: string, body: string, epochSec: number) {
  const abs = join(root, rel);
  await mkdir(join(abs, '..'), { recursive: true });
  await writeFile(abs, body, 'utf-8');
  await utimes(abs, epochSec, epochSec);
}

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), 'mb-gather-'));
});

describe('gatherProjectCore', () => {
  it('reads the three core files, empty string when missing', async () => {
    await touch('context.md', 'ctx', 1000);
    const core = await gatherProjectCore(root);
    expect(core.context).toBe('ctx');
    expect(core.readme).toBe('');
  });
});

describe('gatherUnbuiltSources', () => {
  it('returns only files newer than context.md, newest first', async () => {
    await touch('context.md', 'ctx', 2000);
    await touch('sources/contributors/u/old.md', 'old', 1000);
    await touch('sources/contributors/u/new.md', 'new', 3000);
    await touch('sources/research/newer.md', 'newer', 4000);
    const s = await gatherUnbuiltSources(root);
    expect(s.map((f) => f.path)).toEqual(['sources/research/newer.md', 'sources/contributors/u/new.md']);
  });

  it('includes everything when context.md is missing and skips sidecars', async () => {
    await touch('sources/research/a.md', 'a', 1000);
    await touch('sources/research/a.extracted.md', 'sidecar', 1000);
    const s = await gatherUnbuiltSources(root);
    expect(s.map((f) => f.path)).toEqual(['sources/research/a.md']);
  });
});

describe('parseCitations', () => {
  it('extracts unique trimmed [@path] citations and ignores empty ones', () => {
    const body = 'See [@sources/contributors/u/2026-08-19.md] and [@sources/raw/2026-08-01/x.extracted.md].\n' +
      'Again [@sources/contributors/u/2026-08-19.md]. Empty [@] is skipped.';
    expect(parseCitations(body)).toEqual([
      'sources/contributors/u/2026-08-19.md',
      'sources/raw/2026-08-01/x.extracted.md',
    ]);
  });

  it('returns [] when there are no citations', () => {
    expect(parseCitations('plain [[wikilink]] text')).toEqual([]);
  });
});

describe('gatherResearchPages', () => {
  it('collects cites per page', async () => {
    await touch('sources/research/a.md', '# A\n\nclaim [@sources/contributors/u/2026-01-01.md] [[b]]', 1000);
    await touch('sources/research/b.md', '# B\n\nno cites', 1000);
    const pages = await gatherResearchPages(root);
    const a = pages.find((p) => p.slug === 'a')!;
    const b = pages.find((p) => p.slug === 'b')!;
    expect(a.cites).toEqual(['sources/contributors/u/2026-01-01.md']);
    expect(a.outbound).toEqual(['b']);
    expect(b.cites).toEqual([]);
    expect(b.inboundCount).toBe(1);
  });
});

describe('listResearchSlugs', () => {
  it('returns slugs of *.md files only, and [] when the dir is missing', async () => {
    expect(await listResearchSlugs(root)).toEqual([]);
    await touch('sources/research/deep-work.md', '# DW', 1000);
    await touch('sources/research/b.md', '# B', 1000);
    await touch('sources/research/notes.txt', 'nope', 1000);
    await mkdir(join(root, 'sources', 'research', 'subdir'), { recursive: true });
    expect((await listResearchSlugs(root)).sort()).toEqual(['b', 'deep-work']);
  });
});

describe('listNoteSlugs', () => {
  it('returns note basenames across contributors, ignoring daily files, non-md and nested dirs', async () => {
    expect(await listNoteSlugs(root)).toEqual([]);
    await touch('sources/contributors/u/2026-01-01.md', 'daily, not a note', 1000);
    await touch('sources/contributors/u/notes/deep-work.md', '# DW', 1000);
    await touch('sources/contributors/u/notes/scratch.txt', 'nope', 1000);
    await touch('sources/contributors/v/notes/Attention_Span.md', '# AS', 1000);
    await mkdir(join(root, 'sources', 'contributors', 'v', 'notes', 'sub'), { recursive: true });
    expect((await listNoteSlugs(root)).sort()).toEqual(['Attention_Span', 'deep-work']);
  });
});

describe('findMissingFiles', () => {
  it('returns the paths that are not regular files', async () => {
    await touch('sources/contributors/u/here.md', 'x', 1000);
    const missing = await findMissingFiles(root, [
      'sources/contributors/u/here.md',
      'sources/contributors/u/gone.md',
      'sources/contributors', // a directory is not a citable file
      'sources/contributors/u/gone.md', // duplicates collapse
    ]);
    expect([...missing].sort()).toEqual(['sources/contributors', 'sources/contributors/u/gone.md']);
  });
});

describe('gatherSourceStats', () => {
  it('counts citations from research pages and context.md; uncited sources get 0', async () => {
    await touch('context.md', '# ctx\n\n- fact [@sources/contributors/u/2026-01-02.md]', 5000);
    await touch('sources/contributors/u/2026-01-01.md', 'day one', 1000);
    await touch('sources/contributors/u/2026-01-02.md', 'day two', 2000);
    await touch('sources/contributors/u/notes/idea.md', 'a note', 3000);
    await touch('sources/raw/2026-01-03/abc.extracted.md', 'extracted text', 4000);
    await touch('sources/raw/2026-01-03/abc.pdf', 'binary-ish', 4000);
    await touch('sources/research/r.md', '# R\n\n[@sources/contributors/u/2026-01-01.md] [@sources/contributors/u/2026-01-02.md] [@sources/raw/2026-01-03/abc.extracted.md]', 6000);
    const pages = await gatherResearchPages(root);
    const stats = await gatherSourceStats(root, pages);
    // newest mtime first; raw binary and research pages are not sources
    expect(stats.map((s) => s.path)).toEqual([
      'sources/raw/2026-01-03/abc.extracted.md',
      'sources/contributors/u/notes/idea.md',
      'sources/contributors/u/2026-01-02.md',
      'sources/contributors/u/2026-01-01.md',
    ]);
    const by = Object.fromEntries(stats.map((s) => [s.path, s.citedBy]));
    expect(by['sources/contributors/u/2026-01-02.md']).toBe(2); // research + context.md
    expect(by['sources/contributors/u/2026-01-01.md']).toBe(1);
    expect(by['sources/raw/2026-01-03/abc.extracted.md']).toBe(1);
    expect(by['sources/contributors/u/notes/idea.md']).toBe(0);
    expect(stats.find((s) => s.path === 'sources/contributors/u/2026-01-01.md')!.mtimeMs).toBe(1000 * 1000);
  });

  it('returns [] when no source dirs exist', async () => {
    expect(await gatherSourceStats(root, [])).toEqual([]);
  });
});

function fakeCtx(outputs: string[]): Parameters<typeof completeJson>[0] {
  let call = 0;
  return {
    config: { model: 'fake' },
    getAdapter: () => ({
      chat: (_req: { model: string; messages: ChatMessage[] }): AsyncIterable<ChatChunk> => {
        const text = outputs[Math.min(call++, outputs.length - 1)]!;
        return (async function* () {
          yield { kind: 'delta', text } as ChatChunk;
          yield { kind: 'done', usage: { input_tokens: 1, output_tokens: 1 } } as ChatChunk;
        })();
      },
    }),
  };
}

describe('completeJson', () => {
  const schema = z.object({ x: z.number() });

  it('parses clean JSON', async () => {
    expect(await completeJson(fakeCtx(['{"x": 1}']), { system: 's', user: 'u', schema })).toEqual({ x: 1 });
  });

  it('parses fenced JSON with prose around it', async () => {
    expect(await completeJson(fakeCtx(['Sure! ```json\n{"x": 2}\n``` hope that helps']), { system: 's', user: 'u', schema })).toEqual({ x: 2 });
  });

  it('retries once after invalid output, then succeeds', async () => {
    expect(await completeJson(fakeCtx(['not json at all', '{"x": 3}']), { system: 's', user: 'u', schema })).toEqual({ x: 3 });
  });

  it('throws OpLlmError with raw output after two failures', async () => {
    await expect(completeJson(fakeCtx(['nope', 'still nope']), { system: 's', user: 'u', schema })).rejects.toMatchObject({
      raw: 'still nope',
    });
  });
});
