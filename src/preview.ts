import fs from "node:fs/promises";
import http from "node:http";
import path from "node:path";

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

const port = 4173;
const root = path.join(process.cwd(), "dist");

const server = http.createServer(async (request, response) => {
  const requestPath = request.url === "/" ? "/index.html" : request.url ?? "/index.html";
  const normalized = requestPath.split("?")[0];
  const filePath = path.join(root, normalized);

  try {
    const stat = await fs.stat(filePath).catch(() => null);
    const resolvedFile =
      stat?.isDirectory() ? path.join(filePath, "index.html") : filePath;
    const content = await fs.readFile(resolvedFile);
    response.writeHead(200, {
      "content-type": mimeTypes.get(path.extname(resolvedFile).toLowerCase()) ?? "application/octet-stream"
    });
    response.end(content);
  } catch {
    const fallback = await fs.readFile(path.join(root, "404.html"));
    response.writeHead(404, { "content-type": "text/html; charset=utf-8" });
    response.end(fallback);
  }
});

server.listen(port, () => {
  console.log(`Preview server listening at http://localhost:${port}`);
});
