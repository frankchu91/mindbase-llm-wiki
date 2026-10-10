// apps/server/src/ops/gather.ts
//
// Input gathering for ops recipes. Pure fs reads over a project root; no
// LLM involvement. Caps keep prompts inside local-model context budgets.
import { readFile, readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';

export interface ProjectCore { context: string; indexYaml: string; readme: string }
export interface SourceFile { path: string; body: string }

const MAX_SOURCE_FILES = 30;
const MAX_SOURCE_CHARS = 4_000;

export async function gatherProjectCore(root: string): Promise<ProjectCore> {
  const read = (rel: string) => readFile(join(root, rel), 'utf-8').catch(() => '');
  const [context, indexYaml, readme] = await Promise.all([read('context.md'), read('index.yaml'), read('README.md')]);
  return { context, indexYaml, readme };
}

async function listFilesRec(
  dir: string,
  rel: string,
  keep: (name: string) => boolean = (name) => name.endsWith('.md') && !name.endsWith('.extracted.md'),
): Promise<string[]> {
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return [];
  }
  const out: string[] = [];
  for (const e of entries) {
    if (e.isDirectory()) out.push(...(await listFilesRec(join(dir, e.name), `${rel}/${e.name}`, keep)));
    else if (keep(e.name)) out.push(`${rel}/${e.name}`);
  }
  return out;
}

/**
 * Citation syntax in AI-written pages: `[@<project-relative-path>]`, e.g.
 * `[@sources/contributors/haobing/2026-08-19.md]`. Unique, trimmed, in
 * order of first appearance. No code-fence stripping — keep it simple.
 */
export function parseCitations(body: string): string[] {
  return [...new Set([...body.matchAll(/\[@([^\]\s]+)\]/g)].map((m) => m[1]!.trim()).filter(Boolean))];
}

/**
 * Sources considered "unbuilt": contributor + research files modified after
 * context.md. When context.md is missing, everything counts. Newest first,
 * capped for prompt budget.
 */
export interface ResearchPage {
  path: string;
  slug: string;
  excerpt: string;
  outbound: string[];
  inboundCount: number;
  /** Project-relative source paths cited via `[@path]`. */
  cites: string[];
}

const MAX_LINT_PAGES = 40;
const MAX_EXCERPT_CHARS = 1_200;

/**
 * All research pages with excerpts + the wikilink graph between them
 * (outbound [[slugs]] and inbound counts) — the deterministic evidence
 * the lint recipe hands the LLM so orphan/contradiction hunting isn't
 * left to model recall alone.
 */
export async function gatherResearchPages(root: string): Promise<ResearchPage[]> {
  const rels = (await listFilesRec(join(root, 'sources', 'research'), 'sources/research')).slice(0, MAX_LINT_PAGES);
  const pages = await Promise.all(
    rels.map(async (rel) => {
      const body = await readFile(join(root, rel), 'utf-8').catch(() => '');
      const outbound = [...new Set([...body.matchAll(/\[\[([^\]|#]+)/g)].map((m) => m[1]!.trim()))];
      const slug = (rel.split('/').pop() ?? rel).replace(/\.md$/, '');
      return { path: rel, slug, excerpt: body.slice(0, MAX_EXCERPT_CHARS), outbound, inboundCount: 0, cites: parseCitations(body) };
    }),
  );
  const bySlug = new Map(pages.map((p) => [p.slug, p]));
  for (const p of pages) for (const target of p.outbound) {
    const hit = bySlug.get(target);
    if (hit && hit !== p) hit.inboundCount += 1;
  }
  return pages;
}

/**
 * Slugs of every `sources/research/*.md` page — a cheap directory listing
 * (no bodies) for duplicate-page guards and prompt hints.
 */
export async function listResearchSlugs(root: string): Promise<string[]> {
  const entries = await readdir(join(root, 'sources', 'research'), { withFileTypes: true }).catch(() => []);
  return entries.filter((e) => e.isFile() && e.name.endsWith('.md')).map((e) => e.name.replace(/\.md$/, ''));
}

/**
 * Slugs of every contributor note (`sources/contributors/<user>/notes/*.md`,
 * basename without `.md`) across all contributors. Together with
 * `listResearchSlugs` this is the full set of `[[wikilink]]` targets that
 * resolve, so lint can flag the ones that don't.
 */
export async function listNoteSlugs(root: string): Promise<string[]> {
  const base = join(root, 'sources', 'contributors');
  const users = (await readdir(base, { withFileTypes: true }).catch(() => [])).filter((e) => e.isDirectory());
  const perUser = await Promise.all(
    users.map(async (u) => {
      const entries = await readdir(join(base, u.name, 'notes'), { withFileTypes: true }).catch(() => []);
      return entries.filter((e) => e.isFile() && e.name.endsWith('.md')).map((e) => e.name.replace(/\.md$/, ''));
    }),
  );
  return perUser.flat();
}

/**
 * The subset of project-relative paths that are NOT regular files on disk —
 * one `stat` per unique path. Feeds the lint `broken_citation` check.
 */
export async function findMissingFiles(root: string, rels: Iterable<string>): Promise<Set<string>> {
  const unique = [...new Set(rels)];
  const present = await Promise.all(
    unique.map((rel) => stat(join(root, rel)).then((s) => s.isFile()).catch(() => false)),
  );
  return new Set(unique.filter((_, i) => !present[i]));
}

export interface SourceStat {
  path: string;
  /** Citations from research pages + context.md pointing at this source. */
  citedBy: number;
  mtimeMs: number;
}

const MAX_SOURCE_STATS = 400;

/**
 * Every user-provided source (contributor `*.md` + raw `*.extracted.md`)
 * with a deterministic cited-by count, newest first. Feeds the lint
 * `uncited_source` check and the prompt's SOURCES section.
 */
export async function gatherSourceStats(root: string, pages: ResearchPage[]): Promise<SourceStat[]> {
  const [contributors, raw, context] = await Promise.all([
    listFilesRec(join(root, 'sources', 'contributors'), 'sources/contributors', (n) => n.endsWith('.md')),
    listFilesRec(join(root, 'sources', 'raw'), 'sources/raw', (n) => n.endsWith('.extracted.md')),
    readFile(join(root, 'context.md'), 'utf-8').catch(() => ''),
  ]);
  const counts = new Map<string, number>();
  for (const cited of [...pages.flatMap((p) => p.cites), ...parseCitations(context)]) {
    counts.set(cited, (counts.get(cited) ?? 0) + 1);
  }
  const stats = await Promise.all(
    [...contributors, ...raw].map(async (rel) => ({
      path: rel,
      citedBy: counts.get(rel) ?? 0,
      mtimeMs: await stat(join(root, rel)).then((s) => s.mtimeMs).catch(() => 0),
    })),
  );
  return stats.sort((a, b) => b.mtimeMs - a.mtimeMs).slice(0, MAX_SOURCE_STATS);
}

export async function gatherUnbuiltSources(root: string): Promise<SourceFile[]> {
  const contextMtime = await stat(join(root, 'context.md')).then((s) => s.mtimeMs).catch(() => 0);
  const rels = [
    ...(await listFilesRec(join(root, 'sources', 'contributors'), 'sources/contributors')),
    ...(await listFilesRec(join(root, 'sources', 'research'), 'sources/research')),
  ];
  const withM = await Promise.all(
    rels.map(async (rel) => ({ rel, mtime: await stat(join(root, rel)).then((s) => s.mtimeMs).catch(() => 0) })),
  );
  const fresh = withM.filter((f) => f.mtime > contextMtime).sort((a, b) => b.mtime - a.mtime).slice(0, MAX_SOURCE_FILES);
  return Promise.all(
    fresh.map(async ({ rel }) => ({
      path: rel,
      body: (await readFile(join(root, rel), 'utf-8').catch(() => '')).slice(0, MAX_SOURCE_CHARS),
    })),
  );
}
