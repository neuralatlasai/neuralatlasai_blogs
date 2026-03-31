import fs from "node:fs/promises";
import path from "node:path";

import type { AssetLink } from "./models.js";
import { slugify, toPosix } from "./utils.js";

const MARKDOWN_EXTENSIONS = new Set([".md", ".markdown"]);
const IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".avif"]);

export async function discoverMarkdownFiles(rootDirectory: string): Promise<string[]> {
  const files: string[] = [];

  async function walk(currentDirectory: string): Promise<void> {
    const entries = await fs.readdir(currentDirectory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));

    for (const entry of entries) {
      const absolutePath = path.join(currentDirectory, entry.name);
      if (entry.isDirectory()) {
        await walk(absolutePath);
        continue;
      }

      if (MARKDOWN_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
        files.push(absolutePath);
      }
    }
  }

  await walk(rootDirectory);
  return files;
}

export async function copyTree(sourceDirectory: string, targetDirectory: string): Promise<void> {
  await fs.mkdir(targetDirectory, { recursive: true });
  const entries = await fs.readdir(sourceDirectory, { withFileTypes: true });

  for (const entry of entries) {
    const sourcePath = path.join(sourceDirectory, entry.name);
    const targetPath = path.join(targetDirectory, entry.name);

    if (entry.isDirectory()) {
      await copyTree(sourcePath, targetPath);
      continue;
    }

    await fs.copyFile(sourcePath, targetPath);
  }
}

function toWebSafePath(relativePath: string): string {
  const segments = toPosix(relativePath).split("/");
  const fileName = segments.pop() ?? "asset";
  const extension = path.extname(fileName).toLowerCase();
  const baseName = fileName.slice(0, fileName.length - extension.length);
  const safeBaseName = slugify(baseName);

  return [...segments, `${safeBaseName}${extension}`].join("/");
}

export async function createAssetAliasMap(
  contentRoot: string,
  mirroredContentRoot: string
): Promise<Map<string, string>> {
  const aliases = new Map<string, string>();
  const usedPaths = new Set<string>();

  async function walk(currentDirectory: string): Promise<void> {
    const entries = await fs.readdir(currentDirectory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));

    for (const entry of entries) {
      const absolutePath = path.join(currentDirectory, entry.name);
      if (entry.isDirectory()) {
        await walk(absolutePath);
        continue;
      }

      const extension = path.extname(entry.name).toLowerCase();
      if (MARKDOWN_EXTENSIONS.has(extension)) {
        continue;
      }

      const relativePath = toPosix(path.relative(contentRoot, absolutePath));
      const safePathBase = toWebSafePath(relativePath);
      const safeExtension = path.extname(safePathBase);
      const safeStem = safePathBase.slice(0, safePathBase.length - safeExtension.length);
      let candidatePath = safePathBase;
      let suffix = 1;

      while (usedPaths.has(candidatePath)) {
        candidatePath = `${safeStem}-${suffix}${safeExtension}`;
        suffix += 1;
      }

      usedPaths.add(candidatePath);
      aliases.set(relativePath, candidatePath);

      if (candidatePath === relativePath) {
        continue;
      }

      const mirroredSourcePath = path.join(mirroredContentRoot, relativePath);
      const mirroredAliasPath = path.join(mirroredContentRoot, candidatePath);
      await fs.mkdir(path.dirname(mirroredAliasPath), { recursive: true });
      await fs.copyFile(mirroredSourcePath, mirroredAliasPath);
    }
  }

  await walk(contentRoot);
  return aliases;
}

export async function collectAssetStats(
  sourceDirectory: string,
  contentRoot: string,
  assetAliasMap?: ReadonlyMap<string, string>
): Promise<{
  assetCount: number;
  imageCount: number;
  pdfCount: number;
  pdfFiles: AssetLink[];
}> {
  let assetCount = 0;
  let imageCount = 0;
  let pdfCount = 0;
  const pdfFiles: AssetLink[] = [];

  async function walk(currentDirectory: string): Promise<void> {
    const entries = await fs.readdir(currentDirectory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));

    for (const entry of entries) {
      const absolutePath = path.join(currentDirectory, entry.name);
      if (entry.isDirectory()) {
        await walk(absolutePath);
        continue;
      }

      const extension = path.extname(entry.name).toLowerCase();
      if (MARKDOWN_EXTENSIONS.has(extension)) {
        continue;
      }

      assetCount += 1;
      if (IMAGE_EXTENSIONS.has(extension)) {
        imageCount += 1;
      }

      if (extension === ".pdf") {
        pdfCount += 1;
        const relativePath = toPosix(path.relative(contentRoot, absolutePath));
        pdfFiles.push({
          name: entry.name,
          relativePath: assetAliasMap?.get(relativePath) ?? relativePath
        });
      }
    }
  }

  await walk(sourceDirectory);
  return { assetCount, imageCount, pdfCount, pdfFiles };
}
