import path from "node:path";

const WORD_PATTERN = /[A-Za-z0-9_]+(?:['-][A-Za-z0-9_]+)*/g;

export function toPosix(input: string): string {
  return input.split(path.sep).join("/");
}

export function slugify(input: string): string {
  return input
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/[\s_-]+/g, "-")
    .replace(/^-+|-+$/g, "") || "section";
}

export function createSlugger(): (value: string) => string {
  const counts = new Map<string, number>();

  return (value: string) => {
    const base = slugify(value);
    const seen = counts.get(base) ?? 0;
    counts.set(base, seen + 1);
    return seen === 0 ? base : `${base}-${seen}`;
  };
}

export function humanizeIdentifier(input: string): string {
  return input
    .replace(/\.[^.]+$/u, "")
    .replace(/[_-]+/g, " ")
    .replace(/\btechnical report\b/giu, "")
    .replace(/\breport\b/giu, "")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

export function stripMarkdown(markdown: string): string {
  return markdown
    .replace(/^---[\s\S]*?---\s*/u, "")
    .replace(/```[\s\S]*?```/gu, " ")
    .replace(/`[^`]+`/gu, " ")
    .replace(/\$\$[\s\S]*?\$\$/gu, " ")
    .replace(/\$[^$\n]+\$/gu, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/gu, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/gu, "$1")
    .replace(/<[^>]+>/gu, " ")
    .replace(/^[#>\-\s*]+/gmu, "")
    .replace(/[_*~]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function summarize(markdown: string, maxLength = 220): string {
  const clean = stripMarkdown(markdown);
  return truncateText(clean, maxLength);
}

export function truncateText(text: string, maxLength = 220): string {
  const clean = text.trim();
  if (clean.length <= maxLength) {
    return clean;
  }

  const truncated = clean.slice(0, maxLength);
  const boundary = truncated.lastIndexOf(" ");
  return `${truncated.slice(0, Math.max(boundary, 0)).trim()}...`;
}

export function extractLeadSummary(markdown: string, maxLength = 220): string {
  const withoutFrontmatter = markdown.replace(/^---[\s\S]*?---\s*/u, "").trim();
  const blocks = withoutFrontmatter
    .split(/\n\s*\n/u)
    .map((block) => block.trim())
    .filter(Boolean);

  for (const block of blocks) {
    if (/^#{1,6}\s/u.test(block)) {
      continue;
    }

    if (/^---+$/.test(block)) {
      continue;
    }

    if (/^<img\b/iu.test(block) || /^!\[/u.test(block)) {
      continue;
    }

    const cleaned = stripMarkdown(block);
    if (/^figure\b/iu.test(cleaned) || /^table\b/iu.test(cleaned)) {
      continue;
    }

    if (cleaned.length > 40) {
      return truncateText(cleaned, maxLength);
    }
  }

  return truncateText(stripMarkdown(markdown), maxLength);
}

export function stripLeadingHeading(html: string): string {
  return html
    .replace(/^\s*<h1[^>]*>[\s\S]*?<\/h1>/iu, "")
    .replace(/^\s*<div class="section-divider" aria-hidden="true"><\/div>/iu, "")
    .trim();
}

export function countWords(text: string): number {
  return text.match(WORD_PATTERN)?.length ?? 0;
}

export function estimateReadingMinutes(wordCount: number): number {
  return Math.max(1, Math.ceil(wordCount / 210));
}

export function formatLongDate(date: Date): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric"
  }).format(date);
}

export function relativeHref(fromFile: string, toFile: string): string {
  return toPosix(path.relative(path.dirname(fromFile), toFile)) || ".";
}

export function relativeHrefFromDirectory(fromDirectory: string, toFile: string): string {
  return toPosix(path.relative(fromDirectory, toFile)) || ".";
}

export function escapeHtml(input: string): string {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function versionLabelFromFileName(fileName: string): string | null {
  const match = fileName.match(/v\d+(?:[._-]\d+)*/i);
  if (!match) {
    return null;
  }

  return match[0].replaceAll("_", ".").replaceAll("-", ".");
}

export function versionVector(versionLabel: string | null): number[] {
  if (!versionLabel) {
    return [];
  }

  return versionLabel
    .replace(/^v/i, "")
    .split(/[._-]+/g)
    .map((part) => Number.parseInt(part, 10))
    .filter((value) => Number.isFinite(value));
}

export function compareVersionLabels(left: string | null, right: string | null): number {
  const leftVector = versionVector(left);
  const rightVector = versionVector(right);
  const length = Math.max(leftVector.length, rightVector.length);

  for (let index = 0; index < length; index += 1) {
    const delta = (rightVector[index] ?? 0) - (leftVector[index] ?? 0);
    if (delta !== 0) {
      return delta;
    }
  }

  return 0;
}

export function deriveCollectionTitle(collectionKey: string, titles: string[]): string {
  const candidates = titles
    .flatMap((title) => {
      const prefix = title.match(/^([^:]+):/u)?.[1]?.trim();
      const forPattern = title.match(
        /\bfor\s+([A-Z0-9][A-Za-z0-9 .-]+?)(?:\s+[—-]|$)/u
      )?.[1]?.trim();
      return [prefix, forPattern].filter(
        (candidate): candidate is string =>
          typeof candidate === "string" &&
          !/^end-to-end technical report$/iu.test(candidate) &&
          !/^technical report$/iu.test(candidate)
      );
    })
    .map((candidate) => candidate.trim());
  const uniqueCandidates = [...new Set(candidates)];

  if (uniqueCandidates.length === 1) {
    return uniqueCandidates[0];
  }

  const leaf = collectionKey === "." ? "Technical Reports" : path.posix.basename(collectionKey);
  const leafTitle = humanizeIdentifier(leaf);
  const leafPattern = leafTitle
    .split(/\s+/u)
    .map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("[\\s-]+");
  const numericVariant = titles
    .map((title) => title.match(new RegExp(`\\b${leafPattern}[\\s-]+\\d+(?:\\.\\d+)?`, "iu"))?.[0])
    .find((candidate): candidate is string => typeof candidate === "string");

  return numericVariant ? numericVariant.replace(/-/g, " ") : leafTitle;
}
