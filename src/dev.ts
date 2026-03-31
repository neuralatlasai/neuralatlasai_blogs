import fs from "node:fs";
import fsp from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { spawn } from "node:child_process";

const projectRoot = process.cwd();
const distRoot = path.join(projectRoot, "dist");
const port = Number.parseInt(process.env.PORT ?? "4173", 10) || 4173;

const mimeTypes = new Map<string, string>([
  [".html", "text/html; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml"],
  [".png", "image/png"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
  [".gif", "image/gif"],
  [".webp", "image/webp"],
  [".avif", "image/avif"],
  [".pdf", "application/pdf"],
  [".woff2", "font/woff2"]
]);

let rebuilding = false;
let queuedReason: string | null = null;
let rebuildTimer: NodeJS.Timeout | null = null;
const watchedFileSignatures = new Map<string, string>();

function log(message: string): void {
  const timestamp = new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  }).format(new Date());
  console.log(`[dev ${timestamp}] ${message}`);
}

async function runNodeScript(scriptPath: string, args: string[]): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const child = spawn(process.execPath, [scriptPath, ...args], {
      cwd: projectRoot,
      stdio: "inherit"
    });

    child.once("error", reject);
    child.once("exit", (code) => {
      if (code === 0) {
        resolve();
        return;
      }

      reject(new Error(`${path.basename(scriptPath)} exited with code ${code ?? "unknown"}.`));
    });
  });
}

async function rebuild(reason: string): Promise<void> {
  if (rebuilding) {
    queuedReason = reason;
    return;
  }

  rebuilding = true;
  log(`Rebuilding because ${reason}...`);

  try {
    await runNodeScript(path.join(projectRoot, "node_modules", "typescript", "bin", "tsc"), [
      "-p",
      "tsconfig.json"
    ]);
    await runNodeScript(path.join(projectRoot, "build", "build.js"), []);
    log("Build completed.");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log(`Build failed: ${message}`);
  } finally {
    rebuilding = false;

    if (queuedReason) {
      const nextReason = queuedReason;
      queuedReason = null;
      void rebuild(nextReason);
    }
  }
}

function scheduleRebuild(reason: string): void {
  queuedReason = reason;
  if (rebuildTimer) {
    clearTimeout(rebuildTimer);
  }

  rebuildTimer = setTimeout(() => {
    const nextReason = queuedReason ?? "pending change";
    queuedReason = null;
    void rebuild(nextReason);
  }, 180);
}

async function signatureForPath(filePath: string): Promise<string> {
  try {
    const stat = await fsp.stat(filePath);
    return stat.isDirectory()
      ? `dir:${stat.mtimeMs}`
      : `file:${stat.size}:${stat.mtimeMs}`;
  } catch (error) {
    const systemError = error as NodeJS.ErrnoException;
    if (systemError.code === "ENOENT") {
      return "missing";
    }

    throw error;
  }
}

async function handleWatchedChange(
  watchRoot: string,
  changedFile: string | null,
  isDirectory: boolean
): Promise<void> {
  const absolutePath =
    changedFile && isDirectory
      ? path.resolve(watchRoot, changedFile)
      : watchRoot;
  const relativeLabel = changedFile && isDirectory
    ? `${path.relative(projectRoot, watchRoot)}/${changedFile}`
    : path.relative(projectRoot, watchRoot) || path.basename(watchRoot);
  const normalizedLabel = relativeLabel.replace(/\\/gu, "/");

  if (
    normalizedLabel.startsWith("build/") ||
    normalizedLabel.startsWith("dist/") ||
    normalizedLabel.includes("/.dist-build-")
  ) {
    return;
  }

  try {
    const signature = await signatureForPath(absolutePath);
    const previousSignature = watchedFileSignatures.get(absolutePath);
    if (previousSignature === signature) {
      return;
    }

    watchedFileSignatures.set(absolutePath, signature);
    scheduleRebuild(normalizedLabel);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log(`Watcher stat failed for ${normalizedLabel}: ${message}`);
  }
}

async function handleRequest(
  request: http.IncomingMessage,
  response: http.ServerResponse
): Promise<void> {
  let pathname = "/index.html";

  try {
    const requestUrl = new URL(request.url ?? "/index.html", "http://localhost");
    pathname = requestUrl.pathname === "/" ? "/index.html" : decodeURIComponent(requestUrl.pathname);
  } catch {
    response.writeHead(400, { "content-type": "text/plain; charset=utf-8" });
    response.end("Malformed request path.");
    return;
  }

  const safeRelativePath = path.normalize(pathname).replace(/^([/\\])+/, "");
  const filePath = path.resolve(distRoot, safeRelativePath);
  const distBoundary = `${path.resolve(distRoot)}${path.sep}`;

  if (filePath !== path.resolve(distRoot) && !filePath.startsWith(distBoundary)) {
    response.writeHead(403, { "content-type": "text/plain; charset=utf-8" });
    response.end("Forbidden.");
    return;
  }

  try {
    const stat = await fsp.stat(filePath).catch(() => null);
    const resolvedFile = stat?.isDirectory() ? path.join(filePath, "index.html") : filePath;
    const content = await fsp.readFile(resolvedFile);
    response.writeHead(200, {
      "content-type":
        mimeTypes.get(path.extname(resolvedFile).toLowerCase()) ?? "application/octet-stream"
    });
    response.end(content);
  } catch {
    try {
      const fallback = await fsp.readFile(path.join(distRoot, "404.html"));
      response.writeHead(404, { "content-type": "text/html; charset=utf-8" });
      response.end(fallback);
    } catch {
      response.writeHead(503, { "content-type": "text/plain; charset=utf-8" });
      response.end("Development build is not ready yet.");
    }
  }
}

async function startServer(): Promise<http.Server> {
  for (let candidatePort = port; candidatePort < port + 20; candidatePort += 1) {
    const server = http.createServer((request, response) => {
      void handleRequest(request, response);
    });

    try {
      await new Promise<void>((resolve, reject) => {
        const onError = (error: NodeJS.ErrnoException): void => {
          server.off("listening", onListening);
          reject(error);
        };

        const onListening = (): void => {
          server.off("error", onError);
          resolve();
        };

        server.once("error", onError);
        server.once("listening", onListening);
        server.listen(candidatePort);
      });

      log(`Preview server running at http://localhost:${candidatePort}`);
      return server;
    } catch (error) {
      const systemError = error as NodeJS.ErrnoException;
      if (systemError.code !== "EADDRINUSE") {
        throw error;
      }

      log(`Port ${candidatePort} is busy; trying ${candidatePort + 1}.`);
    }
  }

  throw new Error(`Unable to find a free port in the range ${port}-${port + 19}.`);
}

function installWatcher(target: string): void {
  const absoluteTarget = path.join(projectRoot, target);
  const isDirectory = fs.statSync(absoluteTarget).isDirectory();
  const watcher = fs.watch(
    absoluteTarget,
    { recursive: isDirectory },
    (_eventType, changedFile) => {
      void handleWatchedChange(
        absoluteTarget,
        typeof changedFile === "string" ? changedFile : null,
        isDirectory
      );
    }
  );

  watcher.on("error", (error) => {
    log(`Watcher error on ${target}: ${error.message}`);
  });
}

async function main(): Promise<void> {
  await startServer();
  installWatcher("src");
  installWatcher("techincal_report");
  installWatcher("package.json");
  installWatcher("tsconfig.json");
  await rebuild("startup");
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  log(`Development server failed: ${message}`);
  process.exitCode = 1;
});
