import path from "node:path";

import { siteConfig } from "../site.config.js";
import type { CollectionRecord, DocumentRecord } from "./models.js";
import { escapeHtml, relativeHref } from "./utils.js";

function renderShell(options: {
  title: string;
  description: string;
  outputFile: string;
  outputRoot: string;
  bodyClass: string;
  content: string;
  scriptFile?: string;
}): string {
  const styleHref = relativeHref(options.outputFile, path.join(options.outputRoot, "assets/site.css"));
  const katexHref = relativeHref(
    options.outputFile,
    path.join(options.outputRoot, "assets/vendor/katex.min.css")
  );
  const scriptHref = options.scriptFile ? relativeHref(options.outputFile, options.scriptFile) : null;
  const pageTitle =
    options.title === siteConfig.title
      ? escapeHtml(siteConfig.title)
      : `${escapeHtml(options.title)} | ${escapeHtml(siteConfig.title)}`;

  return `<!DOCTYPE html>
<html lang="${siteConfig.language}">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="description" content="${escapeHtml(options.description)}" />
    <title>${pageTitle}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Bodoni+Moda:opsz,wght@6..96,500;6..96,600;6..96,700&family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap"
    />
    <link rel="stylesheet" href="${styleHref}" />
    <link rel="stylesheet" href="${katexHref}" />
  </head>
  <body class="${options.bodyClass}">
    ${options.content}
    ${
      scriptHref
        ? `<script type="module" src="${scriptHref}"></script>`
        : ""
    }
  </body>
</html>`;
}

function renderArchiveEntry(
  document: DocumentRecord,
  currentOutputFile: string
): string {
  const href = relativeHref(currentOutputFile, document.outputFile);
  const metaParts = [
    document.versionLabel ?? "Primary",
    document.modifiedLabel
  ];

  return `<li class="archive-entry">
  <a class="archive-entry__link" href="${href}">
    <span class="archive-entry__title">${escapeHtml(document.title)}</span>
    <span class="archive-entry__meta">${escapeHtml(metaParts.join(" · "))}</span>
  </a>
</li>`;
}

function renderBrandMark(currentOutputFile: string, outputRoot: string): string {
  const homeHref = relativeHref(currentOutputFile, path.join(outputRoot, "index.html"));
  const logoHref = relativeHref(currentOutputFile, path.join(outputRoot, "assets/271766279.png"));

  return `<a class="brand-mark" href="${homeHref}" aria-label="${escapeHtml(siteConfig.title)} archive">
  <img class="brand-mark__logo" src="${logoHref}" alt="${escapeHtml(siteConfig.title)}" width="40" height="40" />
  <span class="brand-mark__text">${escapeHtml(siteConfig.shortTitle)}</span>
</a>`;
}

function renderResourceList(collection: CollectionRecord, currentOutputFile: string, outputRoot: string): string {
  if (collection.pdfFiles.length === 0) {
    return "";
  }

  return `<section class="resource-panel">
  <h2>Supporting PDFs</h2>
  <ul class="resource-list">
    ${collection.pdfFiles
      .map((asset) => {
        const href = relativeHref(
          currentOutputFile,
          path.join(outputRoot, siteConfig.contentRoot, asset.relativePath)
        );
        return `<li><a href="${href}">${escapeHtml(asset.name)}</a></li>`;
      })
      .join("")}
  </ul>
</section>`;
}

export function renderDocumentPage(options: {
  document: DocumentRecord;
  collection: CollectionRecord;
  outputRoot: string;
}): string {
  const { document, collection, outputRoot } = options;
  const homeHref = relativeHref(document.outputFile, path.join(outputRoot, "index.html"));
  const collectionHref = relativeHref(document.outputFile, collection.outputFile);
  const sourceHref = relativeHref(
    document.outputFile,
    path.join(path.dirname(document.outputFile), document.sourceFileName)
  );
  const sectionHeadings = document.headings.filter((heading) => heading.depth > 1);
  const metaParts = [
    collection.title,
    document.versionLabel ?? "Primary",
    document.modifiedLabel,
    `${document.readingMinutes} min`,
    `${document.wordCount.toLocaleString("en-US")} words`
  ];

  const content = `<main class="page-shell page-shell--article">
  ${renderBrandMark(document.outputFile, outputRoot)}
  ${
    sectionHeadings.length > 0
      ? `<div class="current-section" data-current-section aria-hidden="true">
    <div class="current-section__inner">
      <span class="current-section__title" data-current-section-title>${escapeHtml(sectionHeadings[0].text)}</span>
    </div>
  </div>`
      : ""
  }
  <section class="paper-layout">
    <article class="paper">
      <nav class="breadcrumbs" aria-label="Breadcrumb">
        <a href="${homeHref}">Archive</a>
        <span>/</span>
        <a href="${collectionHref}">${escapeHtml(collection.title)}</a>
      </nav>
      <header class="paper-header">
        <h1>${escapeHtml(document.title)}</h1>
        <div class="paper-meta">${escapeHtml(metaParts.join(" · "))}</div>
        <div class="paper-links">
          <a href="${collectionHref}">Collection</a>
          <a href="${sourceHref}">Markdown</a>
        </div>
      </header>
      <div class="article-content prose">
        ${document.bodyHtml}
      </div>
    </article>
  </section>
</main>`;

  return renderShell({
    title: document.title,
    description: document.summary,
    outputFile: document.outputFile,
    outputRoot,
    bodyClass: "page page--article",
    content,
    scriptFile: path.join(outputRoot, "assets/article.js")
  });
}

export function renderCollectionPage(options: {
  collection: CollectionRecord;
  outputRoot: string;
}): string {
  const { collection, outputRoot } = options;
  const homeHref = relativeHref(collection.outputFile, path.join(outputRoot, "index.html"));
  const collectionMeta = [
    `${collection.documents.length} documents`,
    collection.imageCount > 0 ? `${collection.imageCount} images` : null,
    collection.pdfCount > 0 ? `${collection.pdfCount} PDFs` : null
  ].filter((value): value is string => typeof value === "string");

  const content = `<main class="page-shell page-shell--collection">
  ${renderBrandMark(collection.outputFile, outputRoot)}
  <section class="archive-panel">
    <nav class="breadcrumbs" aria-label="Breadcrumb">
      <a href="${homeHref}">Archive</a>
      <span>/</span>
      <span>${escapeHtml(collection.title)}</span>
    </nav>
    <header class="archive-panel__header">
      <h1>${escapeHtml(collection.title)}</h1>
      <p>${escapeHtml(collectionMeta.join(" · "))}</p>
    </header>
    <ol class="archive-list archive-list--panel">
      ${collection.documents
        .map((document) => renderArchiveEntry(document, collection.outputFile))
        .join("")}
    </ol>
    ${renderResourceList(collection, collection.outputFile, outputRoot)}
  </section>
</main>`;

  return renderShell({
    title: collection.title,
    description: collection.description,
    outputFile: collection.outputFile,
    outputRoot,
    bodyClass: "page page--collection",
    content,
    scriptFile: path.join(outputRoot, "assets/home.js")
  });
}

export function renderHomePage(options: {
  collections: CollectionRecord[];
  outputFile: string;
  outputRoot: string;
}): string {
  const { collections, outputFile, outputRoot } = options;

  const content = `<main class="page-shell page-shell--home">
  <section class="archive-index">
    <header class="archive-index__header">
      <div>
        ${renderBrandMark(outputFile, outputRoot)}
        <h1>Technical Reports</h1>
      </div>
      <label class="archive-search">
        <span class="visually-hidden">Filter reports</span>
        <input data-search-input type="search" placeholder="Filter reports" />
      </label>
    </header>
    <div class="archive-groups">
      ${collections
        .map((collection) => {
          const href = relativeHref(outputFile, collection.outputFile);
          const groupMeta = [
            `${collection.documents.length} documents`,
            collection.pdfCount > 0 ? `${collection.pdfCount} PDFs` : null
          ].filter((value): value is string => typeof value === "string");

          return `<section class="archive-group" data-filter-card data-search="${escapeHtml(
            collection.searchText.toLowerCase()
          )}">
  <header class="archive-group__header">
    <h2><a href="${href}">${escapeHtml(collection.title)}</a></h2>
    <p>${escapeHtml(groupMeta.join(" · "))}</p>
  </header>
  <ol class="archive-list">
    ${collection.documents
      .map((document) => renderArchiveEntry(document, outputFile))
      .join("")}
  </ol>
</section>`;
        })
        .join("")}
    </div>
    <section class="empty-state" data-empty-state hidden>
      <p>No matching report collection.</p>
    </section>
  </section>
</main>`;

  return renderShell({
    title: siteConfig.title,
    description: siteConfig.description,
    outputFile,
    outputRoot,
    bodyClass: "page page--home",
    content,
    scriptFile: path.join(outputRoot, "assets/home.js")
  });
}

export function renderNotFoundPage(outputFile: string, outputRoot: string): string {
  const homeHref = relativeHref(outputFile, path.join(outputRoot, "index.html"));
  const content = `<main class="page-shell page-shell--not-found">
  <section class="not-found">
    ${renderBrandMark(outputFile, outputRoot)}
    <nav class="breadcrumbs" aria-label="Breadcrumb">
      <a href="${homeHref}">Archive</a>
    </nav>
    <h1>Page not found</h1>
    <p><a href="${homeHref}">Return to the archive</a></p>
  </section>
</main>`;

  return renderShell({
    title: "Page not found",
    description: "The requested report page does not exist.",
    outputFile,
    outputRoot,
    bodyClass: "page page--not-found",
    content
  });
}
