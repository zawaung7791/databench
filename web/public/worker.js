// Browser worker. Two modes:
//  - pool member: processes one range of the shared buffers (multi-core run)
//  - "runLocal": runs the whole benchmark in this one worker (fallback when the
//    page isn't cross-origin isolated, so SharedArrayBuffer isn't available)
importScripts("/bench.js");
const handle = self.Bench.workerHandler((m) => self.postMessage(m));
self.onmessage = (e) => {
  const m = e.data;
  if (m.type !== "runLocal") return handle(m);
  self.Bench.run(m.rows, self.Bench.localPool()).then(
    (out) => self.postMessage({ type: "done", ok: true, data: out }),
    (err) => self.postMessage({ type: "done", ok: false, error: String(err) })
  );
};
