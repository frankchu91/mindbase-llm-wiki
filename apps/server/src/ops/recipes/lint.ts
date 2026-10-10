// apps/server/src/ops/recipes/lint.ts
//
// Lint emits findings only — it never writes to the wiki. Findings are
// cached to artifacts/lint/<date>.json by the runner so the Health view
// persists across reloads.
import { z } from 'zod';
import type { ProjectCore, ResearchPage, SourceStat } from '../gather';

export const FINDING_KINDS = [
  'contradiction',
  'stale',
  'orphan',
  'missing_page',
  'missing_link',
  'gap',
  'question',
  // Computed deterministically by the runner, never requested from the model.
  'unsourced_page',
  'uncited_source',
  'broken_citation',
  'broken_link',
  'near_duplicate',
] as const;

/** Kinds the runner computes in code; any the model emits anyway are dropped. */
export const DETERMINISTIC_KINDS: ReadonlySet<Finding['kind']> = new Set<Finding['kind']>([
  'unsourced_page',
  'uncited_source',
  'broken_citation',
  'broken_link',
  'near_duplicate',
]);

export const evidenceSchema = z.object({
  page: z.string().min(1),
  quote: z.string().min(8).max(300),
});

export const findingSchema = z.object({
  kind: z.enum(FINDING_KINDS),
  pages: z.array(z.string().min(1)).max(6),
  detail: z.string().min(1).max(600),
  evidence: z.array(evidenceSchema).max(4).optional(),
});
export type Finding = z.infer<typeof findingSchema>;

export const lintSchema = z.object({
  findings: z.array(findingSchema).max(20),
});
export type LintOutput = z.infer<typeof lintSchema>;

const SYSTEM = `You are MindBase's wiki health checker. You READ the project
and report problems — you never propose edits. Respond with ONLY JSON:
{ "findings": [ { "kind": "...", "pages": ["path-or-slug", ...], "detail": "...",
                  "evidence": [ { "page": "path-or-slug", "quote": "verbatim text" } ] }, ... ] }
kind is EXACTLY one of:
  contradiction — two pages/claims that disagree; quote both sides in detail
  stale         — a claim likely superseded by newer content
  orphan        — a page nothing links to (use the provided inbound counts)
  missing_page  — a concept referenced repeatedly but with no page of its own
  missing_link  — two clearly related pages that don't reference each other
  gap           — a question the project raises but never answers
  question      — a suggested next question worth investigating
Evidence: for "contradiction" findings, "evidence" is REQUIRED — one entry per
conflicting page, with "quote" copied VERBATIM (no paraphrase, no ellipsis,
8-300 characters) from that page. For "stale", include at least one verbatim
quote. Findings whose quotes cannot be found verbatim in the named page will
be discarded. Evidence is optional for other kinds.
Do NOT emit kinds "unsourced_page", "uncited_source", "broken_citation",
"broken_link" or "near_duplicate" — uncited pages/sources, broken citations,
broken wikilinks and near-duplicate pages are all computed by the system from
the link and citation data below. In particular, "missing_link" is ONLY for a
one-way cross-reference between two pages that BOTH exist; never use it for a
link whose target does not exist.
Rules: every finding must cite the specific page paths in "pages"; detail is
1-3 sentences, concrete, quoting the conflicting text when relevant; do not
invent pages that were not provided; return at most 12 findings, the most
important first; an empty findings array is a valid answer for a healthy wiki.`;

const MAX_PROMPT_SOURCES = 30;

export function lintPrompt(input: { core: ProjectCore; pages: ResearchPage[]; sources?: SourceStat[] }): { system: string; user: string } {
  const graph = input.pages
    .map((p) => {
      const parts = [`inbound: ${p.inboundCount}`];
      if (p.outbound.length) parts.push(`links to: ${p.outbound.join(', ')}`);
      if (p.cites.length) parts.push(`cites: ${p.cites.join(', ')}`);
      return `- ${p.path} (${parts.join('; ')})`;
    })
    .join('\n');
  const sources = (input.sources ?? [])
    .slice(0, MAX_PROMPT_SOURCES)
    .map((s) => `- ${s.path} (cited by ${s.citedBy})`)
    .join('\n');
  const excerpts = input.pages.map((p) => `--- ${p.path}\n${p.excerpt}`).join('\n\n');
  return {
    system: SYSTEM,
    user: `context.md (curated truth — cite it as "context.md"):\n${input.core.context || '(empty)'}\n\nLINK GRAPH (deterministic — trust these inbound counts):\n${graph || '(no research pages)'}\n\nSOURCES (deterministic cited-by counts):\n${sources || '(no sources)'}\n\nPAGE EXCERPTS:\n${excerpts || '(none)'}\n\nReport findings as JSON.`,
  };
}
