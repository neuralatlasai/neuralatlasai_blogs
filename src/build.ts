import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { siteConfig } from "./site.config.js";
import { collectAssetStats, copyTree, createAssetAliasMap, discoverMarkdownFiles } from "./lib/content.js";
import type { CollectionRecord, DocumentRecord } from "./lib/models.js";
import { createMarkdownEngine } from "./lib/markdown.js";
import {
  compareVersionLabels,
  countWords,
  deriveCollectionTitle,
  estimateReadingMinutes,
  formatLongDate,
  humanizeIdentifier,
  toPosix,
  versionLabelFromFileName
} from "./lib/utils.js";
import {
  renderCollectionPage,
  renderDocumentPage,
  renderHomePage,
  renderNotFoundPage
} from "./lib/template.js";
import { validateRenderedHtml } from "./lib/validation.js";

async function copyStaticAssets(projectRoot: string, outputRoot: string): Promise<void> {
  const compiledRoot = path.dirname(fileURLToPath(import.meta.url));
  const assetRoot = path.join(outputRoot, "assets");
  const vendorRoot = path.join(assetRoot, "vendor");

  await fs.mkdir(vendorRoot, { recursive: true });
  await copyTree(path.join(projectRoot, "src/assets"), assetRoot);
  await fs.copyFile(path.join(projectRoot, "src/styles/site.css"), path.join(assetRoot, "site.css"));
  await fs.copyFile(path.join(compiledRoot, "client/article.js"), path.join(assetRoot, "article.js"));
  await fs.copyFile(path.join(compiledRoot, "client/home.js"), path.join(assetRoot, "home.js"));
  await fs.copyFile(
    path.join(projectRoot, "node_modules/katex/dist/katex.min.css"),
    path.join(vendorRoot, "katex.min.css")
  );
  await copyTree(
    path.join(projectRoot, "node_modules/katex/dist/fonts"),
    path.join(vendorRoot, "fonts")
  );
}

async function ensureDirectoryForFile(filePath: string): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
}

async function removeDirectory(directoryPath: string): Promise<void> {
  await fs.rm(directoryPath, {
    recursive: true,
    force: true,
    maxRetries: 8,
    retryDelay: 120
  });
}

function createTemporaryOutputRoot(projectRoot: string): string {
  return path.join(
    projectRoot,
    `.dist-build-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  );
}

async function removeStaleTemporaryOutputRoots(projectRoot: string): Promise<void> {
  const entries = await fs.readdir(projectRoot, {
    withFileTypes: true
  });

  const isProcessAlive = (pid: number): boolean => {
    try {
      process.kill(pid, 0);
      return true;
    } catch {
      return false;
    }
  };

  await Promise.all(
    entries
      .filter((entry) => {
        if (!entry.isDirectory() || !entry.name.startsWith(".dist-build-")) {
          return false;
        }

        const match = entry.name.match(/^\.dist-build-(\d+)-/u);
        if (!match) {
          return true;
        }

        const pid = Number.parseInt(match[1], 10);
        return Number.isNaN(pid) || !isProcessAlive(pid);
      })
      .map((entry) => removeDirectory(path.join(projectRoot, entry.name)))
  );
}

function compareDocuments(left: DocumentRecord, right: DocumentRecord): number {
  const versionDelta = compareVersionLabels(left.versionLabel, right.versionLabel);
  if (versionDelta !== 0) {
    return versionDelta;
  }

  const modifiedDelta = right.modifiedAt.getTime() - left.modifiedAt.getTime();
  if (modifiedDelta !== 0) {
    return modifiedDelta;
  }

  return left.title.localeCompare(right.title);
}

async function buildDocuments(projectRoot: string, outputRoot: string): Promise<{
  documents: DocumentRecord[];
  collections: CollectionRecord[];
}> {
  const contentRoot = path.join(projectRoot, siteConfig.contentRoot);
  const mirroredContentRoot = path.join(outputRoot, siteConfig.contentRoot);
  const markdownFiles = await discoverMarkdownFiles(contentRoot);
  const markdownEngine = await createMarkdownEngine();
  const assetAliasMap = await createAssetAliasMap(contentRoot, mirroredContentRoot);
  const documents: DocumentRecord[] = [];

  function rewriteDocumentAssetUrls(bodyHtml: string, inputFile: string): string {
    const documentDirectory = path.dirname(inputFile);
    const documentDirectoryRelative = toPosix(path.relative(contentRoot, documentDirectory)) || ".";

    return bodyHtml.replace(
      /\b(href|src|poster)=("([^"]*)"|'([^']*)')/giu,
      (match, attribute: string, _quoted: string, doubleQuotedValue?: string, singleQuotedValue?: string) => {
        const rawValue = doubleQuotedValue ?? singleQuotedValue ?? "";
        if (
          rawValue.length === 0 ||
          rawValue.startsWith("#") ||
          /^(?:[a-z][a-z0-9+.-]*:)?\/\//iu.test(rawValue) ||
          /^(?:mailto|tel|data|javascript):/iu.test(rawValue)
        ) {
          return match;
        }

        const [pathAndQuery, hash = ""] = rawValue.split("#", 2);
        const [resourcePath, query = ""] = pathAndQuery.split("?", 2);
        const resolvedAssetPath = path.resolve(documentDirectory, decodeURIComponent(resourcePath));
        if (path.relative(contentRoot, resolvedAssetPath).startsWith("..")) {
          return match;
        }

        const assetRelativePath = toPosix(path.relative(contentRoot, resolvedAssetPath));
        const aliasedRelativePath = assetAliasMap.get(assetRelativePath);
        if (!aliasedRelativePath) {
          return match;
        }

        const fromDirectory = path.posix.join(
          "/",
          documentDirectoryRelative === "." ? "" : documentDirectoryRelative
        );
        const toAsset = path.posix.join("/", aliasedRelativePath);
        const rewrittenPath = path.posix.relative(fromDirectory, toAsset) || path.posix.basename(toAsset);
        const rewrittenValue = `${encodeURI(rewrittenPath)}${query ? `?${query}` : ""}${hash ? `#${hash}` : ""}`;
        return `${attribute}="${rewrittenValue}"`;
      }
    );
  }

  for (const inputFile of markdownFiles) {
    const relativeInputPath = path.relative(projectRoot, inputFile);
    const outputFile = path.join(outputRoot, relativeInputPath).replace(/\.md$/iu, ".html");
    const source = await fs.readFile(inputFile, "utf8");
    const rendered = markdownEngine.render(source);
    const stats = await fs.stat(inputFile);
    const collectionKey = toPosix(path.relative(contentRoot, path.dirname(inputFile))) || ".";
    const title =
      rendered.frontmatter.title ??
      rendered.titleFromContent ??
      humanizeIdentifier(path.basename(inputFile));
    const summary =
      rendered.frontmatter.summary ??
      rendered.frontmatter.description ??
      rendered.excerpt;
    const wordCount = rendered.wordCount || countWords(source);

    documents.push({
      id: toPosix(path.relative(contentRoot, inputFile)),
      inputFile,
      outputFile,
      relativeOutputPath: toPosix(path.relative(outputRoot, outputFile)),
      collectionKey,
      title,
      summary,
      bodyHtml: rewriteDocumentAssetUrls(rendered.html, inputFile),
      headings: rendered.headings,
      wordCount,
      readingMinutes: estimateReadingMinutes(wordCount),
      modifiedAt: stats.mtime,
      modifiedLabel: formatLongDate(stats.mtime),
      versionLabel: versionLabelFromFileName(path.basename(inputFile)),
      sourceFileName: path.basename(inputFile),
      sourceRelativePath: toPosix(path.relative(mirroredContentRoot, path.join(outputRoot, relativeInputPath))),
      searchText: `${title} ${summary} ${collectionKey}`
    });
  }

  const grouped = new Map<string, DocumentRecord[]>();
  for (const document of documents) {
    const existing = grouped.get(document.collectionKey) ?? [];
    existing.push(document);
    grouped.set(document.collectionKey, existing);
  }

  const collections: CollectionRecord[] = [];
  for (const [collectionKey, collectionDocuments] of grouped) {
    collectionDocuments.sort(compareDocuments);
    const sourceDirectory =
      collectionKey === "."
        ? contentRoot
        : path.join(contentRoot, collectionKey);
    const title = deriveCollectionTitle(
      collectionKey,
      collectionDocuments.map((document) => document.title)
    );
    const description =
      collectionDocuments.find((document) => document.summary.trim().length > 0)?.summary ??
      `Structured technical report collection for ${title}.`;
    const outputFile =
      collectionKey === "."
        ? path.join(outputRoot, siteConfig.contentRoot, "index.html")
        : path.join(outputRoot, siteConfig.contentRoot, collectionKey, "index.html");
    const assetStats = await collectAssetStats(sourceDirectory, contentRoot, assetAliasMap);

    collections.push({
      key: collectionKey,
      title,
      description,
      sourceDirectory,
      outputFile,
      relativeOutputPath: toPosix(path.relative(outputRoot, outputFile)),
      documents: collectionDocuments,
      assetCount: assetStats.assetCount,
      imageCount: assetStats.imageCount,
      pdfCount: assetStats.pdfCount,
      pdfFiles: assetStats.pdfFiles,
      searchText: `${title} ${description} ${collectionKey} ${collectionDocuments
        .map((document) => document.searchText)
        .join(" ")}`
    });
  }

  collections.sort((left, right) => left.title.localeCompare(right.title));
  return { documents, collections };
}

async function writeGeneratedPages(
  outputRoot: string,
  collections: CollectionRecord[]
): Promise<void> {
  for (const collection of collections) {
    await ensureDirectoryForFile(collection.outputFile);
    const collectionHtml = renderCollectionPage({
      collection,
      outputRoot
    });
    validateRenderedHtml(collection.outputFile, outputRoot, collectionHtml);
    await fs.writeFile(
      collection.outputFile,
      collectionHtml,
      "utf8"
    );

    for (const document of collection.documents) {
      await ensureDirectoryForFile(document.outputFile);
      const documentHtml = renderDocumentPage({
        document,
        collection,
        outputRoot
      });
      validateRenderedHtml(document.outputFile, outputRoot, documentHtml);
      await fs.writeFile(
        document.outputFile,
        documentHtml,
        "utf8"
      );
    }
  }

  const homeFile = path.join(outputRoot, "index.html");
  const homeHtml = renderHomePage({
    collections,
    outputFile: homeFile,
    outputRoot
  });
  validateRenderedHtml(homeFile, outputRoot, homeHtml);
  await fs.writeFile(
    homeFile,
    homeHtml,
    "utf8"
  );

  const notFoundFile = path.join(outputRoot, "404.html");
  const notFoundHtml = renderNotFoundPage(notFoundFile, outputRoot);
  validateRenderedHtml(notFoundFile, outputRoot, notFoundHtml);
  await fs.writeFile(notFoundFile, notFoundHtml, "utf8");
  await fs.writeFile(path.join(outputRoot, ".nojekyll"), "", "utf8");
}

async function main(): Promise<void> {
  const projectRoot = process.cwd();
  const outputRoot = path.join(projectRoot, siteConfig.outputRoot);
  const temporaryOutputRoot = createTemporaryOutputRoot(projectRoot);
  const contentRoot = path.join(projectRoot, siteConfig.contentRoot);

  try {
    await removeStaleTemporaryOutputRoots(projectRoot);
    await removeDirectory(temporaryOutputRoot);
    await fs.mkdir(temporaryOutputRoot, { recursive: true });
    await copyTree(contentRoot, path.join(temporaryOutputRoot, siteConfig.contentRoot));
    await copyStaticAssets(projectRoot, temporaryOutputRoot);

    const { collections } = await buildDocuments(projectRoot, temporaryOutputRoot);
    await writeGeneratedPages(temporaryOutputRoot, collections);

    await removeDirectory(outputRoot);
    await fs.rename(temporaryOutputRoot, outputRoot);

    console.log(
      `Built ${collections.reduce((sum, collection) => sum + collection.documents.length, 0)} documents across ${
        collections.length
      } collections into ${siteConfig.outputRoot}.`
    );
  } finally {
    await removeDirectory(temporaryOutputRoot);
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
