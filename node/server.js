const http = require("http");
const os = require("os");
const path = require("path");
const { Worker } = require("worker_threads");
const { run, createPool } = require("./bench");

const PORT = 8000;
const MAX_ROWS = parseInt(process.env.MAX_ROWS || "2000000", 10);
const WORKERS = parseInt(process.env.BENCH_WORKERS || "0", 10) || os.availableParallelism();

// Start the worker pool once, so thread start-up isn't counted in the timings.
const pool = createPool(
  Array.from({ length: WORKERS }, () => {
    const w = new Worker(path.join(__dirname, "worker.js"));
    return { post: (m) => w.postMessage(m), listen: (fn) => w.on("message", fn) };
  })
);

let busy = false;

function send(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store" });
  res.end(JSON.stringify(body));
}

http
  .createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost");
    if (url.pathname === "/health") return send(res, 200, { ok: true, workers: WORKERS });
    if (url.pathname !== "/bench") return send(res, 404, { error: "Not found" });
    if (busy) return send(res, 429, { error: "A JavaScript benchmark is already running. Try again when it finishes." });

    const asked = parseInt(url.searchParams.get("rows") || "1000000", 10) || 1000000;
    const n = Math.max(1000, Math.min(asked, MAX_ROWS));
    busy = true;
    try {
      const out = await run(n, pool);
      out.language = "JavaScript";
      out.runtime = `Node.js ${process.version}`;
      send(res, 200, out);
    } catch (err) {
      send(res, 500, { error: String(err) });
    } finally {
      busy = false;
    }
  })
  .listen(PORT, () => console.log(`node bench listening on ${PORT} with ${WORKERS} worker threads`));
