import fs from "node:fs";
import path from "node:path";

import { toPosix } from "./utils.js";

type ValidationIssue = {
  file: string;
  line: number;
  message: string;
  snippet: string;
};

function lineNumberAt(input: string, index: number): number {
  let line = 1;

  for (let cursor = 0; cursor < index; cursor += 1) {
    if (input[cursor] === "\n") {
      line += 1;
    }
  }

  return line;
}

function snippetAt(input: string, index: number, length: number): string {
  return input
    .slice(Math.max(0, index - 80), Math.min(input.length, index + length + 80))
    .replace(/\s+/gu, " ")
    .trim();
}

function visibleTextSegments(html: string): Array<{
  text: string;
  index: number;
}> {
  const maskedHtml = html
    .replace(/<pre\b[\s\S]*?<\/pre>/giu, (segment) => " ".repeat(segment.length))
    .replace(/<code\b[\s\S]*?<\/code>/giu, (segment) => " ".repeat(segment.length))
    .replace(/<annotation\b[\s\S]*?<\/annotation>/giu, (segment) => " ".repeat(segment.length));

  const segments: Array<{
    text: string;
    index: number;
  }> = [];

  for (const match of maskedHtml.matchAll(/>([^<]+)</gu)) {
    if (typeof match.index !== "number") {
      continue;
    }

    const text = match[1];
    if (text.trim().length === 0) {
      continue;
    }

    segments.push({
      text,
      index: match.index + 1
    });
  }

  return segments;
}

export function validateRenderedHtml(outputFile: string, outputRoot: string, html: string): void {
  const issues: ValidationIssue[] = [];
  const relativeFile = toPosix(path.relative(outputRoot, outputFile));

  const katexErrorPattern = /katex-error/gu;
  for (const match of html.matchAll(katexErrorPattern)) {
    if (typeof match.index !== "number") {
      continue;
    }

    issues.push({
      file: relativeFile,
      line: lineNumberAt(html, match.index),
      message: "KaTeX rendering error leaked into output HTML.",
      snippet: snippetAt(html, match.index, match[0].length)
    });
  }

  const rawMathPattern = /(?<!\\)\$[^$\n]{1,160}(?<!\\)\$/gu;
  for (const segment of visibleTextSegments(html)) {
    for (const match of segment.text.matchAll(rawMathPattern)) {
      if (typeof match.index !== "number") {
        continue;
      }

      const matchIndex = segment.index + match.index;
      issues.push({
        file: relativeFile,
        line: lineNumberAt(html, matchIndex),
        message: "Unrendered inline math delimiter leaked into visible HTML content.",
        snippet: snippetAt(html, matchIndex, match[0].length)
      });
    }

    const leakedMarkupPattern =
      /(?:&lt;\/?(?:span|section|math|mrow|mtable|mtr|mtd|annotation|svg|path)\b|data-label=|class=&quot;katex|katex-display)/gu;
    for (const match of segment.text.matchAll(leakedMarkupPattern)) {
      if (typeof match.index !== "number") {
        continue;
      }

      const matchIndex = segment.index + match.index;
      issues.push({
        file: relativeFile,
        line: lineNumberAt(html, matchIndex),
        message: "Escaped HTML or KaTeX markup leaked into visible article text.",
        snippet: snippetAt(html, matchIndex, match[0].length)
      });
    }
  }

  const localResourcePattern = /\b(href|src|poster)=("([^"]*)"|'([^']*)')/giu;
  for (const match of html.matchAll(localResourcePattern)) {
    if (typeof match.index !== "number") {
      continue;
    }

    const attribute = match[1];
    const resource = match[3] ?? match[4] ?? "";
    if (
      resource.length === 0 ||
      resource.startsWith("#") ||
      /^(?:[a-z][a-z0-9+.-]*:)?\/\//iu.test(resource) ||
      /^(?:mailto|tel|data|javascript):/iu.test(resource)
    ) {
      continue;
    }

    const [pathAndQuery] = resource.split("#", 1);
    const [resourcePath] = pathAndQuery.split("?", 1);
    const extension = path.extname(resourcePath).toLowerCase();
    if (attribute === "href" && (extension.length === 0 || extension === ".html")) {
      continue;
    }

    const resolvedPath = path.resolve(path.dirname(outputFile), decodeURIComponent(resourcePath));
    const outputBoundary = `${path.resolve(outputRoot)}${path.sep}`;
    if (resolvedPath !== path.resolve(outputRoot) && !resolvedPath.startsWith(outputBoundary)) {
      continue;
    }

    if (!fs.existsSync(resolvedPath)) {
      issues.push({
        file: relativeFile,
        line: lineNumberAt(html, match.index),
        message: "Local resource reference points to a file that does not exist in output.",
        snippet: snippetAt(html, match.index, match[0].length)
      });
    }
  }

  if (issues.length === 0) {
    return;
  }

  const summary = issues
    .slice(0, 12)
    .map(
      (issue) =>
        `${issue.file}:${issue.line} ${issue.message} Snippet: ${issue.snippet}`
    )
    .join("\n");

  throw new Error(`Render validation failed.\n${summary}`);
}
