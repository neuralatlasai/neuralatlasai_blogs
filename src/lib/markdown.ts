import matter from "gray-matter";
import katex from "katex";
import MarkdownIt from "markdown-it";
import attrs from "markdown-it-attrs";
import container from "markdown-it-container";
import deflist from "markdown-it-deflist";
import footnote from "markdown-it-footnote";
import sub from "markdown-it-sub";
import sup from "markdown-it-sup";
import taskLists from "markdown-it-task-lists";
import { createHighlighter, type HighlighterGeneric } from "shiki";

import type { DocumentFrontmatter, Heading, RenderedMarkdown } from "./models.js";
import {
  countWords,
  createSlugger,
  escapeHtml,
  extractLeadSummary,
  slugify,
  stripLeadingHeading
} from "./utils.js";

const semanticBlocks = [
  ["note", "Note"],
  ["definition", "Definition"],
  ["theorem", "Theorem"],
  ["insight", "Insight"],
  ["warning", "Warning"],
  ["implementation", "Implementation"],
  ["observation", "Observation"],
  ["result", "Result"],
  ["benchmark", "Benchmark"],
  ["comparison", "Comparison"],
  ["open", "Open Problem"]
] as const;

const alertMap = new Map<string, string>([
  ["note", "note"],
  ["info", "note"],
  ["important", "note"],
  ["tip", "insight"],
  ["definition", "definition"],
  ["theorem", "theorem"],
  ["warning", "warning"],
  ["caution", "warning"],
  ["danger", "warning"],
  ["implementation", "implementation"],
  ["insight", "insight"],
  ["observation", "observation"],
  ["result", "result"],
  ["benchmark", "benchmark"],
  ["comparison", "comparison"],
  ["open", "open"],
  ["question", "open"]
]);

type MarkdownToken = {
  type: string;
  tag: string;
  info: string;
  content: string;
  nesting: number;
  children: MarkdownToken[] | null;
  attrGet(name: string): string | null;
  attrSet(name: string, value: string): void;
  attrJoin(name: string, value: string): void;
};

type RenderRule = (
  tokens: any[],
  index: number,
  options: any,
  env: any,
  self: any
) => string;

const AUDIO_ASSET_PATTERN = /\.(?:mp3|wav|ogg|m4a|aac|flac)(?:[?#][^"]*)?$/iu;
const VIDEO_ASSET_PATTERN = /\.(?:mp4|webm|mov|m4v|ogv)(?:[?#][^"]*)?$/iu;

function isExternalUrl(input: string): boolean {
  return (
    /^(?:[a-z][a-z0-9+.-]*:)?\/\//iu.test(input) ||
    /^(?:mailto|tel|data|javascript):/iu.test(input)
  );
}

function encodeDocumentUrl(input: string): string {
  const trimmed = input.trim();
  if (trimmed.length === 0 || trimmed.startsWith("#") || isExternalUrl(trimmed)) {
    return trimmed;
  }

  const hashIndex = trimmed.indexOf("#");
  const resource = hashIndex === -1 ? trimmed : trimmed.slice(0, hashIndex);
  const hash = hashIndex === -1 ? "" : trimmed.slice(hashIndex);
  return `${encodeURI(resource)}${hash}`;
}

function rewriteHtmlResourceAttributes(html: string): string {
  return html.replace(/\b(href|src|poster)=("([^"]*)"|'([^']*)')/giu, (_match, attribute, _quoted, doubleQuotedValue, singleQuotedValue) => {
    const value = typeof doubleQuotedValue === "string" ? doubleQuotedValue : singleQuotedValue;
    const encoded = encodeDocumentUrl(value ?? "");
    return `${attribute}="${escapeHtml(encoded)}"`;
  });
}

function registerSemanticContainers(markdown: MarkdownIt): void {
  for (const [kind, label] of semanticBlocks) {
    markdown.use(container, kind, {
      validate: (params: string) => params.trim().startsWith(kind),
      render: (tokens: MarkdownToken[], index: number) => {
        const token = tokens[index];
        const info = token.info.trim();
        const title = info.slice(kind.length).trim();

        if (token.nesting === 1) {
          const titleHtml = title
            ? `<div class="semantic-block__title">${escapeHtml(title)}</div>`
            : "";
          return `<section class="semantic-block semantic-block--${kind}" data-block-kind="${kind}"><div class="semantic-block__eyebrow">${label}</div>${titleHtml}<div class="semantic-block__body">\n`;
        }

        return "</div></section>\n";
      }
    });
  }
}

function transformAlerts(markdown: string): string {
  const lines = markdown.split(/\r?\n/u);
  const output: string[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    const match = lines[index].match(/^>\s*\[!([A-Z]+)\]\s*(.*)$/iu);
    if (!match) {
      output.push(lines[index]);
      continue;
    }

    const kind = alertMap.get(match[1].toLowerCase()) ?? "note";
    const title = match[2].trim();
    output.push(`::: ${kind}${title ? ` ${title}` : ""}`);
    index += 1;

    while (index < lines.length && /^>\s?/u.test(lines[index])) {
      output.push(lines[index].replace(/^>\s?/u, ""));
      index += 1;
    }

    output.push(":::");
    index -= 1;
  }

  return output.join("\n");
}

function normalizeFrontmatter(rawFrontmatter: Record<string, unknown>): DocumentFrontmatter {
  const tags = Array.isArray(rawFrontmatter.tags)
    ? rawFrontmatter.tags.filter((value): value is string => typeof value === "string")
    : undefined;

  return {
    title: typeof rawFrontmatter.title === "string" ? rawFrontmatter.title : undefined,
    description:
      typeof rawFrontmatter.description === "string" ? rawFrontmatter.description : undefined,
    summary: typeof rawFrontmatter.summary === "string" ? rawFrontmatter.summary : undefined,
    tags,
    draft: typeof rawFrontmatter.draft === "boolean" ? rawFrontmatter.draft : undefined
  };
}

function escapeRegex(input: string): string {
  return input.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function decodeHtmlAttribute(input: string): string {
  return input
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&amp;", "&");
}

function protectFencedCodeBlocks(markdown: string): {
  text: string;
  segments: Map<string, string>;
} {
  const lines = markdown.split(/\r?\n/u);
  const output: string[] = [];
  const segments = new Map<string, string>();
  let buffer: string[] = [];
  let fenceCharacter = "";
  let fenceLength = 0;
  let segmentIndex = 0;

  for (const line of lines) {
    if (fenceCharacter.length === 0) {
      const openMatch = line.match(/^ {0,3}([`~]{3,})(.*)$/u);
      if (!openMatch) {
        output.push(line);
        continue;
      }

      fenceCharacter = openMatch[1][0];
      fenceLength = openMatch[1].length;
      buffer = [line];
      continue;
    }

    buffer.push(line);
    const closePattern = new RegExp(
      `^ {0,3}${escapeRegex(fenceCharacter.repeat(fenceLength))}${escapeRegex(fenceCharacter)}*\\s*$`,
      "u"
    );

    if (!closePattern.test(line)) {
      continue;
    }

    const token = `@@CODE_BLOCK_${segmentIndex}@@`;
    segments.set(token, buffer.join("\n"));
    output.push(token);
    buffer = [];
    fenceCharacter = "";
    fenceLength = 0;
    segmentIndex += 1;
  }

  if (buffer.length > 0) {
    output.push(...buffer);
  }

  return {
    text: output.join("\n"),
    segments
  };
}

function protectInlineCode(line: string): {
  text: string;
  segments: Map<string, string>;
} {
  const segments = new Map<string, string>();
  let result = "";
  let index = 0;
  let segmentIndex = 0;

  while (index < line.length) {
    if (line[index] !== "`") {
      result += line[index];
      index += 1;
      continue;
    }

    let tickCount = 1;
    while (line[index + tickCount] === "`") {
      tickCount += 1;
    }

    const delimiter = "`".repeat(tickCount);
    const closeIndex = line.indexOf(delimiter, index + tickCount);
    if (closeIndex === -1) {
      result += line[index];
      index += 1;
      continue;
    }

    const token = `@@INLINE_CODE_${segmentIndex}@@`;
    segments.set(token, line.slice(index, closeIndex + tickCount));
    result += token;
    index = closeIndex + tickCount;
    segmentIndex += 1;
  }

  return {
    text: result,
    segments
  };
}

function restoreProtectedSegments(text: string, segments: Map<string, string>): string {
  let restored = text;

  for (const [token, value] of segments) {
    restored = restored.replaceAll(token, value);
  }

  return restored;
}

function normalizeMathSource(input: string): string | null {
  const trimmed = input.trim();
  if (trimmed.length === 0) {
    return null;
  }

  return trimmed.replace(/(?<!\\)%/gu, "\\%");
}

function serializeMathAttribute(input: string): string {
  return escapeHtml(input.replace(/\s+/gu, " ").trim());
}

function toPlainMathLabel(input: string): string {
  return input
    .replace(/\\text\{([^}]*)\}/gu, "$1")
    .replace(/\\(?:mathbf|mathbb|mathcal|mathrm|mathit|operatorname)\{([^}]*)\}/gu, "$1")
    .replace(/\\times/gu, "x")
    .replace(/\\cdot/gu, "·")
    .replace(/\\rightarrow|\\Rightarrow/gu, "->")
    .replace(/\\leftarrow|\\Leftarrow/gu, "<-")
    .replace(/\\leftrightarrow|\\Leftrightarrow/gu, "<->")
    .replace(/\\leq/gu, "<=")
    .replace(/\\geq/gu, ">=")
    .replace(/\\neq/gu, "!=")
    .replace(/\\sim|\\approx|\\simeq|\\asymp/gu, "~")
    .replace(/\\pm/gu, "+/-")
    .replace(/\\mp/gu, "-/+")
    .replace(/\\to/gu, "to")
    .replace(/\\_/gu, "_")
    .replace(/\\%/gu, "%")
    .replace(/\\left|\\right|\\bigl|\\bigr|\\Bigl|\\Bigr/gu, "")
    .replace(/[\\{}]/gu, "")
    .replace(/\s+/gu, " ")
    .trim();
}

function makeMathFragmentTableSafe(fragment: string): string {
  // Markdown tables split on raw pipe characters before HTML parsing, so
  // inline math like $|\mathcal{V}|$ must not emit literal pipes at this stage.
  return fragment.replace(/\|/gu, "&#124;");
}

function sanitizeRenderedMathHtml(rendered: string): string {
  // markdown-it still tokenizes inline HTML boundaries. When KaTeX emits a raw
  // backslash as a text node immediately before a closing tag, markdown treats
  // the backslash as escaping the `<` of that closing tag and leaks markup into
  // visible text. Encode those operator glyphs before injecting the fragment.
  return rendered.replace(/>(\\+)(?=<\/)/gu, (_match, slashes: string) => {
    return `>${"&#92;".repeat(slashes.length)}`;
  });
}

function renderMathFragment(input: string, displayMode: boolean): string {
  const normalized = normalizeMathSource(input);
  if (!normalized) {
    return "";
  }

  const label = toPlainMathLabel(normalized);
  const serializedTex = serializeMathAttribute(normalized);
  const serializedLabel = serializeMathAttribute(label);
  const rendered = katex.renderToString(normalized, {
    displayMode,
    throwOnError: false,
    strict: "ignore"
  });

  if (rendered.includes("katex-error")) {
    const fallback = `<span class="math-fallback">${escapeHtml(label || normalized)}</span>`;
    const fragment = displayMode
      ? `\n<section class="math-block math-block--fallback"><eqn data-tex="${serializedTex}" data-label="${serializedLabel}">${fallback}</eqn></section>\n`
      : fallback;
    return makeMathFragmentTableSafe(fragment);
  }

  const element = displayMode ? "eqn" : "eq";
  const wrapperClass = displayMode ? "math-block" : "";
  const body = `<${element} data-tex="${serializedTex}" data-label="${serializedLabel}">${sanitizeRenderedMathHtml(
    rendered
  )}</${element}>`;

  const fragment = displayMode ? `\n<section class="${wrapperClass}">${body}</section>\n` : body;
  return makeMathFragmentTableSafe(fragment);
}

function renderDelimitedDisplayMathBlocks(
  markdown: string,
  openDelimiter: string,
  closeDelimiter: string
): string {
  let output = "";
  let index = 0;

  while (index < markdown.length) {
    const startIndex = markdown.indexOf(openDelimiter, index);
    if (startIndex === -1) {
      output += markdown.slice(index);
      break;
    }

    const endIndex = markdown.indexOf(closeDelimiter, startIndex + openDelimiter.length);
    if (endIndex === -1) {
      output += markdown.slice(index);
      break;
    }

    output += markdown.slice(index, startIndex);
    output += renderMathFragment(
      markdown.slice(startIndex + openDelimiter.length, endIndex),
      true
    );
    index = endIndex + closeDelimiter.length;
  }

  return output;
}

function renderBareMathEnvironments(markdown: string): string {
  return markdown.replace(
    /(^|\n\s*\n)([ \t]*\\begin\{([a-zA-Z*]+)\}[\s\S]*?[ \t]*\\end\{\3\}[ \t]*)(?=\n\s*\n|$)/gu,
    (_match, prefix: string, block: string) => `${prefix}${renderMathFragment(block, true)}`
  );
}

function findClosingDisplayDelimiter(input: string, startIndex: number): number {
  for (let index = startIndex; index < input.length - 1; index += 1) {
    if (input[index] === "$" && input[index + 1] === "$" && input[index - 1] !== "\\") {
      return index;
    }
  }

  return -1;
}

function renderDisplayMathBlocks(markdown: string): string {
  let output = "";
  let index = 0;

  while (index < markdown.length) {
    const startIndex = markdown.indexOf("$$", index);
    if (startIndex === -1) {
      output += markdown.slice(index);
      break;
    }

    if (startIndex > 0 && markdown[startIndex - 1] === "\\") {
      output += markdown.slice(index, startIndex + 1);
      index = startIndex + 1;
      continue;
    }

    const endIndex = findClosingDisplayDelimiter(markdown, startIndex + 2);
    if (endIndex === -1) {
      output += markdown.slice(index);
      break;
    }

    output += markdown.slice(index, startIndex);
    output += renderMathFragment(markdown.slice(startIndex + 2, endIndex), true);
    index = endIndex + 2;
  }

  return output;
}

function isStandaloneMathOperator(input: string): boolean {
  return /^\\(?:sim|approx|simeq|asymp|pm|mp|times|cdot|leq|geq|neq|ll|gg|rightarrow|leftarrow|Rightarrow|Leftarrow|to)$/u.test(
    input.trim()
  );
}

function renderEscapedInlineMathInLine(line: string): string {
  let output = "";
  let index = 0;

  while (index < line.length) {
    const startIndex = line.indexOf("\\(", index);
    if (startIndex === -1) {
      output += line.slice(index);
      break;
    }

    const endIndex = line.indexOf("\\)", startIndex + 2);
    if (endIndex === -1) {
      output += line.slice(index);
      break;
    }

    output += line.slice(index, startIndex);
    const mathSource = line.slice(startIndex + 2, endIndex);
    output += renderMathFragment(mathSource, false) || `\\(${mathSource}\\)`;
    index = endIndex + 2;
  }

  return output;
}

function renderInlineMathInLine(line: string): string {
  let output = "";
  let index = 0;

  while (index < line.length) {
    if (line[index] !== "$" || line[index + 1] === "$" || (index > 0 && line[index - 1] === "\\")) {
      output += line[index];
      index += 1;
      continue;
    }

    let closeIndex = -1;
    for (let cursor = index + 1; cursor < line.length; cursor += 1) {
      if (line[cursor] === "$" && line[cursor - 1] !== "\\" && line[cursor + 1] !== "$") {
        closeIndex = cursor;
        break;
      }
    }

    if (closeIndex === -1) {
      output += line[index];
      index += 1;
      continue;
    }

    const mathSource = line.slice(index + 1, closeIndex);
    const rendered = renderMathFragment(mathSource, false);
    output += rendered || `$${mathSource}$`;

    if (isStandaloneMathOperator(mathSource) && /[0-9A-Za-z]/u.test(line[closeIndex + 1] ?? "")) {
      output += " ";
    }

    index = closeIndex + 1;
  }

  return output;
}

function preprocessMath(markdown: string): string {
  const protectedFences = protectFencedCodeBlocks(markdown);
  const withDollarDisplayMath = renderDisplayMathBlocks(protectedFences.text);
  const withBracketDisplayMath = renderDelimitedDisplayMathBlocks(
    withDollarDisplayMath,
    "\\[",
    "\\]"
  );
  const withEnvironmentMath = renderBareMathEnvironments(withBracketDisplayMath);
  const processedLines = withEnvironmentMath.split("\n").map((line) => {
    const protectedInlineCode = protectInlineCode(line);
    const withEscapedInlineMath = renderEscapedInlineMathInLine(protectedInlineCode.text);
    const withInlineMath = renderInlineMathInLine(withEscapedInlineMath);
    return restoreProtectedSegments(withInlineMath, protectedInlineCode.segments);
  });

  return restoreProtectedSegments(processedLines.join("\n"), protectedFences.segments);
}

function stripTrailingHeadingId(rawText: string): string {
  return rawText.replace(/\s*\{#.+\}\s*$/u, "").trim();
}

function extractHeadingText(inlineToken: MarkdownToken | undefined): string {
  if (!inlineToken) {
    return "";
  }

  if (!inlineToken.children || inlineToken.children.length === 0) {
    return stripTrailingHeadingId(inlineToken.content);
  }

  const text = inlineToken.children
    .map((child) => {
      if (child.type === "html_inline") {
        const labelMatch = child.content.match(/\sdata-label="([^"]*)"/u);
        return labelMatch ? decodeHtmlAttribute(labelMatch[1]) : "";
      }

      return child.content;
    })
    .join("");

  return stripTrailingHeadingId(text.replace(/\s+/gu, " ").trim());
}

function stripHeadingIdFromInlineToken(inlineToken: MarkdownToken | undefined): void {
  if (!inlineToken) {
    return;
  }

  inlineToken.content = stripTrailingHeadingId(inlineToken.content);
  for (const child of inlineToken.children ?? []) {
    if (child.type === "text") {
      child.content = stripTrailingHeadingId(child.content);
    }
  }
}

function configureRenderer(markdown: MarkdownIt, highlighter: HighlighterGeneric<any, any>): void {
  const defaultHeadingOpen: RenderRule =
    markdown.renderer.rules.heading_open ??
    ((tokens, index, options, _env, self) => self.renderToken(tokens, index, options));
  const defaultHeadingClose: RenderRule =
    markdown.renderer.rules.heading_close ??
    ((tokens, index, options, _env, self) => self.renderToken(tokens, index, options));
  const defaultLinkOpen: RenderRule =
    markdown.renderer.rules.link_open ??
    ((tokens, index, options, _env, self) => self.renderToken(tokens, index, options));
  const defaultImage: RenderRule =
    markdown.renderer.rules.image ??
    ((tokens, index, options, _env, self) => self.renderToken(tokens, index, options));

  markdown.renderer.rules.heading_open = ((tokens, index, options, env, self) => {
    const token = tokens[index];
    token.attrJoin("class", "article-heading");
    return defaultHeadingOpen(tokens, index, options, env, self);
  }) as RenderRule;

  markdown.renderer.rules.heading_close = ((tokens, index, options, env, self) => {
    const openingToken = tokens[index - 2];
    const id = openingToken?.attrGet("id");
    const permalink = id
      ? `<a class="heading-anchor" href="#${escapeHtml(id)}" aria-label="Link to section">#</a>`
      : "";
    return `${permalink}${defaultHeadingClose(tokens, index, options, env, self)}`;
  }) as RenderRule;

  markdown.renderer.rules.link_open = ((tokens, index, options, env, self) => {
    const token = tokens[index];
    let href = token.attrGet("href") ?? "";

    if (/^(https?:)?\/\//u.test(href)) {
      token.attrSet("target", "_blank");
      token.attrSet("rel", "noreferrer noopener");
    } else if (/\.md(#.*)?$/iu.test(href)) {
      href = href.replace(/\.md(?=#|$)/iu, ".html");
    }

    token.attrSet("href", encodeDocumentUrl(href));
    return defaultLinkOpen(tokens, index, options, env, self);
  }) as RenderRule;

  markdown.renderer.rules.image = ((tokens, index, options, env, self) => {
    const token = tokens[index];
    const source = token.attrGet("src");
    if (source) {
      token.attrSet("src", encodeDocumentUrl(source));
    }
    token.attrSet("loading", "lazy");
    token.attrSet("decoding", "async");
    token.attrJoin("class", "article-image");
    return defaultImage(tokens, index, options, env, self);
  }) as RenderRule;

  markdown.renderer.rules.fence = ((tokens, index) => {
    const token = tokens[index];
    const info = token.info.trim();
    const requestedLanguage = info.split(/\s+/u)[0] || "text";
    const isPlainTextFence = new Set(["text", "plaintext", "algorithm", "pseudocode", "pseudo"]).has(
      requestedLanguage
    );
    if (requestedLanguage === "math") {
      return renderMathFragment(token.content.trim(), true);
    }

    const language = highlighter
      .getLoadedLanguages()
      .includes((isPlainTextFence ? "text" : requestedLanguage) as never)
      ? (isPlainTextFence ? "text" : requestedLanguage)
      : "text";
    const normalizedContent =
      isPlainTextFence
        ? token.content.replace(/\n[ \t]*\n+/gu, "\n").trimEnd()
        : token.content.trimEnd();
    const highlighted = highlighter.codeToHtml(normalizedContent, {
      lang: language,
      themes: {
        light: "github-light",
        dark: "github-dark-dimmed"
      }
    });
    const header = isPlainTextFence
      ? `<figcaption class="code-frame__header code-frame__header--plain"><button class="code-frame__copy" type="button">Copy</button></figcaption>`
      : `<figcaption class="code-frame__header"><span class="code-frame__language">${escapeHtml(
          requestedLanguage
        )}</span><button class="code-frame__copy" type="button">Copy</button></figcaption>`;

    return `<figure class="code-frame" data-language="${escapeHtml(
      requestedLanguage
    )}"${isPlainTextFence ? ' data-frame-kind="plain"' : ""}>${header}${highlighted}</figure>`;
  }) as RenderRule;

  markdown.renderer.rules.table_open = () => '<div class="table-scroll"><table>';
  markdown.renderer.rules.table_close = () => "</table></div>";
}

function assignHeadingIds(tokens: MarkdownToken[]): Heading[] {
  const slugger = createSlugger();
  const headings: Heading[] = [];

  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (token.type !== "heading_open") {
      continue;
    }

    const inlineToken = tokens[index + 1];
    const title = extractHeadingText(inlineToken);
    const explicitId = inlineToken?.content.match(/\{#(.+)\}\s*$/u)?.[1];
    const id = explicitId ? slugify(explicitId) : slugger(title);
    stripHeadingIdFromInlineToken(inlineToken);
    token.attrSet("id", id);
    headings.push({
      depth: Number.parseInt(token.tag.replace("h", ""), 10),
      id,
      text: title
    });
  }

  return headings;
}

function wrapRawTables(html: string): string {
  return html.replace(/<table\b[\s\S]*?<\/table>/giu, (table, offset, input) => {
    const before = input.slice(Math.max(0, offset - 80), offset);
    return before.includes('<div class="table-scroll">') ? table : `<div class="table-scroll">${table}</div>`;
  });
}

function decorateEmbeddedHtml(html: string): string {
  return rewriteHtmlResourceAttributes(
    html
      .replace(/<img\b(?![^>]*\bclass=)([^>]*)>/giu, '<img class="article-image"$1>')
      .replace(/<img\b(?![^>]*\bloading=)([^>]*)>/giu, '<img loading="lazy" decoding="async"$1>')
    .replace(/<audio\b(?![^>]*\bclass=)([^>]*)>/giu, '<audio class="article-audio"$1>')
    .replace(/<audio\b(?![^>]*\bcontrols\b)([^>]*)>/giu, '<audio controls preload="metadata"$1>')
    .replace(/<video\b(?![^>]*\bclass=)([^>]*)>/giu, '<video class="article-video"$1>')
    .replace(/<video\b(?![^>]*\bcontrols\b)([^>]*)>/giu, '<video controls preload="metadata" playsinline$1>')
    .replace(/<iframe\b(?![^>]*\bclass=)([^>]*)>/giu, '<iframe class="article-iframe" loading="lazy"$1>')
  );
}

function renderStandaloneMediaLinks(html: string): string {
  return html
    .replace(
      /<p>\s*<a\b[^>]*href="([^"]+\.(?:mp3|wav|ogg|m4a|aac|flac)(?:[?#][^"]*)?)"[^>]*>([\s\S]*?)<\/a>\s*<\/p>/giu,
      (_match, href: string, label: string) => {
        const caption = label.trim() && label.trim() !== href
          ? `<figcaption class="article-media__caption">${label.trim()}</figcaption>`
          : "";
        return `<figure class="article-media article-media--audio"><audio class="article-audio" controls preload="metadata" src="${escapeHtml(
          encodeDocumentUrl(href)
        )}"></audio>${caption}</figure>`;
      }
    )
    .replace(
      /<p>\s*<a\b[^>]*href="([^"]+\.(?:mp4|webm|mov|m4v|ogv)(?:[?#][^"]*)?)"[^>]*>([\s\S]*?)<\/a>\s*<\/p>/giu,
      (_match, href: string, label: string) => {
        const caption = label.trim() && label.trim() !== href
          ? `<figcaption class="article-media__caption">${label.trim()}</figcaption>`
          : "";
        return `<figure class="article-media article-media--video"><video class="article-video" controls preload="metadata" playsinline src="${escapeHtml(
          encodeDocumentUrl(href)
        )}"></video>${caption}</figure>`;
      }
    );
}

function wrapFigureCaptions(html: string): string {
  return html.replace(
    /(<img\b[^>]*>|<(?:audio|video|iframe)\b[\s\S]*?<\/(?:audio|video|iframe)>)\s*<p class="figure-caption">([\s\S]*?)<\/p>/giu,
    (_match, media: string, caption: string) => {
      return `<figure class="article-figure">${media}<figcaption class="figure-caption">${caption}</figcaption></figure>`;
    }
  );
}

function polishArticleHtml(html: string): string {
  const normalizedHtml = html
    .replace(/<a\b([^>]*)href="([^"]+)"([^>]*)>([\s\S]*?)<\/a>/giu, (match, before, href, after, label) => {
      if (!AUDIO_ASSET_PATTERN.test(href) && !VIDEO_ASSET_PATTERN.test(href)) {
        return match;
      }

      return `<a${before}href="${href}"${after} data-media-link="true">${label}</a>`;
    })
    .replace(/<p>\s*(<section class="math-block[\s\S]*?<\/section>)\s*<\/p>/gu, "$1")
    .replace(/<p><em>(Figure\.[\s\S]*?)<\/em><\/p>/gu, '<p class="figure-caption"><em>$1</em></p>')
    .replace(/<p><em>(Table\.[\s\S]*?)<\/em><\/p>/gu, '<p class="figure-caption"><em>$1</em></p>')
    .replace(/<hr\s*\/?>/gu, '<div class="section-divider" aria-hidden="true"></div>')
    .replace(/\sdata-media-link="true"/giu, "");

  const withStandaloneMedia = renderStandaloneMediaLinks(normalizedHtml);
  const withWrappedTables = wrapRawTables(withStandaloneMedia);
  const withDecoratedHtml = decorateEmbeddedHtml(withWrappedTables);
  return wrapFigureCaptions(withDecoratedHtml);
}

export async function createMarkdownEngine(): Promise<{
  render(markdownSource: string): RenderedMarkdown;
}> {
  const highlighter = await createHighlighter({
    themes: ["github-light", "github-dark-dimmed"],
    langs: [
      "text",
      "plaintext",
      "bash",
      "ts",
      "tsx",
      "js",
      "jsx",
      "json",
      "yaml",
      "html",
      "css",
      "md",
      "markdown",
      "python",
      "rust",
      "cpp",
      "sql"
    ]
  });

  const markdown = new MarkdownIt({
    html: true,
    linkify: true,
    typographer: true,
    breaks: false
  });

  markdown.use(deflist);
  markdown.use(footnote);
  markdown.use(sub);
  markdown.use(sup);
  markdown.use(attrs);
  markdown.use(taskLists, { enabled: true, label: true, labelAfter: true });

  registerSemanticContainers(markdown);
  configureRenderer(markdown, highlighter);

  return {
    render(markdownSource: string): RenderedMarkdown {
      const parsed = matter(markdownSource);
      const frontmatter = normalizeFrontmatter(parsed.data);
      const transformed = preprocessMath(transformAlerts(parsed.content));
      const tokens = markdown.parse(transformed, {});
      const headings = assignHeadingIds(tokens);
      const html = stripLeadingHeading(
        polishArticleHtml(markdown.renderer.render(tokens, markdown.options, {}))
      );
      const excerpt =
        frontmatter.summary ?? frontmatter.description ?? extractLeadSummary(parsed.content);
      const wordCount = countWords(parsed.content);

      return {
        frontmatter,
        titleFromContent: headings.find((heading) => heading.depth === 1)?.text ?? null,
        html,
        headings,
        excerpt,
        wordCount
      };
    }
  };
}
