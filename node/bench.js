// Shared benchmark core, multi-core version.
// Used by Node.js (worker_threads) and the browser (Web Workers). The main thread
// splits the rows into one range per worker; workers fill and process shared memory,
// and the main thread combines the partial results (and merges the sorted runs).
// Same mulberry32 data (seed 42) as the Rust and Python engines.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Bench = factory();
})(typeof self !== "undefined" ? self : this, function () {
  const SEED = 42;          // main stream: category, region, value, quantity
  const SEED_CUST = 1001;   // which customer each row belongs to
  const SEED_PROD = 2002;   // which product code each row has
  const SEED_TIER = 3003;   // each customer's tier
  const K = 0x6d2b79f5;
  const N_CAT = 20;
  const N_REG = 5;
  const N_CUST = 100000;
  const N_PROD = 5000;
  const DISC = [0, 0.05, 0.1, 0.2]; // discount per customer tier
  const custId = (c) => c * 7919 + 13; // sparse ids, so a real hash lookup is needed

  // mulberry32 output for counter j (the generator state is just seed + j*K)
  function rand(j, seed = SEED) {
    const a = (seed + Math.imul(j, K)) | 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // ---------------------------------------------------------------- memory
  function allocate(n, shared) {
    const B = shared ? SharedArrayBuffer : ArrayBuffer;
    return {
      n,
      cat: new B(n), reg: new B(n), val: new B(8 * n),
      qty: new B(4 * n), idx: new B(4 * n), srt: new B(8 * n),
      cref: new B(4 * n), cid: new B(4 * N_CUST), ctier: new B(N_CUST),
    };
  }
  function views(b) {
    return {
      n: b.n,
      cat: new Uint8Array(b.cat), reg: new Uint8Array(b.reg), val: new Float64Array(b.val),
      qty: new Uint32Array(b.qty), idx: new Uint32Array(b.idx), srt: new Float64Array(b.srt),
      cref: new Uint32Array(b.cref), cid: new Uint32Array(b.cid), ctier: new Uint8Array(b.ctier),
      sku: null, skuStart: 0, // product codes are strings, so each worker keeps its own range
    };
  }

  // ------------------------------------------------- kernels (one range each)
  const OPS = {
    gen(v, s, e, arg) {
      v.sku = new Array(e - s);
      v.skuStart = s;
      for (let i = s; i < e; i++) {
        const j = 4 * i;
        v.cat[i] = (rand(j + 1) * N_CAT) | 0;
        v.reg[i] = (rand(j + 2) * N_REG) | 0;
        v.val[i] = rand(j + 3) * 1000;
        v.qty[i] = 1 + ((rand(j + 4) * 10) | 0);
        v.cref[i] = custId((rand(i + 1, SEED_CUST) * N_CUST) | 0);
        v.sku[i - s] = "SKU-" + String((rand(i + 1, SEED_PROD) * N_PROD) | 0).padStart(4, "0");
      }
      for (let c = arg.cs; c < arg.ce; c++) {      // this worker's share of the customer table
        v.cid[c] = custId(c);
        v.ctier[c] = (rand(c + 1, SEED_TIER) * 4) | 0;
      }
      return null;
    },
    filter(v, s, e) {
      let count = 0, revenue = 0;
      for (let i = s; i < e; i++) {
        if (v.val[i] > 500 && v.qty[i] >= 5) { count++; revenue += v.val[i] * v.qty[i]; }
      }
      return { count, revenue };
    },
    groupCat(v, s, e) {
      const sum = new Array(N_CAT).fill(0), cnt = new Array(N_CAT).fill(0);
      for (let i = s; i < e; i++) { sum[v.cat[i]] += v.val[i]; cnt[v.cat[i]]++; }
      return { sum, cnt };
    },
    groupRegCat(v, s, e) {
      const rev = new Array(N_REG * N_CAT).fill(0);
      for (let i = s; i < e; i++) rev[v.reg[i] * N_CAT + v.cat[i]] += v.val[i] * v.qty[i];
      return rev;
    },
    join(v, s, e) {
      const tierOf = new Map();                    // hash index on the customer table
      for (let c = 0; c < N_CUST; c++) tierOf.set(v.cid[c], v.ctier[c]);
      const net = [0, 0, 0, 0];
      for (let i = s; i < e; i++) {
        const t = tierOf.get(v.cref[i]);
        net[t] += v.val[i] * v.qty[i] * (1 - DISC[t]);
      }
      return net;
    },
    groupCust(v, s, e) {
      const rev = new Map();
      for (let i = s; i < e; i++) {
        const k = v.cref[i];
        rev.set(k, (rev.get(k) || 0) + v.val[i] * v.qty[i]);
      }
      return rev;
    },
    text(v, s, e) {
      let count = 0;
      const sums = new Map();
      for (let i = s; i < e; i++) {
        const code = v.sku[i - v.skuStart];
        if (code.includes("42")) { count++; sums.set(code, (sums.get(code) || 0) + v.val[i]); }
      }
      return { count, sums };
    },
    sortIdx(v, s, e) {
      const val = v.val;
      for (let i = s; i < e; i++) v.idx[i] = i;
      v.idx.subarray(s, e).sort((x, y) => val[y] - val[x]);
      return null;
    },
    sum(v, s, e) {
      let t = 0;
      for (let i = s; i < e; i++) t += v.val[i];
      return t;
    },
    sqdev(v, s, e, mean) {
      let t = 0;
      for (let i = s; i < e; i++) { const d = v.val[i] - mean; t += d * d; }
      return t;
    },
    sortVals(v, s, e) {
      v.srt.set(v.val.subarray(s, e), s);
      v.srt.subarray(s, e).sort(); // typed arrays sort numerically
      return null;
    },
  };
  function exec(v, job) { return OPS[job.op](v, job.s, job.e, job.arg); }

  // ------------------------------------------------------------------ pools
  // adapters: [{ post(msg), listen(fn) }] wrapping Node or browser workers
  function createPool(adapters) {
    let seq = 0;
    const pending = new Map();
    adapters.forEach((a) => a.listen((m) => {
      const p = pending.get(m.id);
      if (!p) return;
      pending.delete(m.id);
      m.error ? p.reject(new Error(m.error)) : p.resolve(m.result);
    }));
    const call = (i, msg) => new Promise((resolve, reject) => {
      const id = ++seq;
      pending.set(id, { resolve, reject });
      adapters[i].post({ ...msg, id });
    });
    return {
      size: adapters.length,
      shared: true,
      init: (buffers) => Promise.all(adapters.map((_, i) => call(i, { type: "init", buffers }))),
      run: (jobs) => Promise.all(jobs.map((job, i) => call(i % adapters.length, { type: "job", job }))),
    };
  }
  // Everything in the current thread (fallback when shared memory isn't available)
  function localPool() {
    let v = null;
    return {
      size: 1,
      shared: false,
      init: async (b) => { v = views(b); },
      run: async (jobs) => jobs.map((j) => exec(v, j)),
    };
  }
  // Message handler for the worker side
  function workerHandler(post) {
    let v = null;
    return (m) => {
      try {
        if (m.type === "init") { v = views(m.buffers); post({ id: m.id, result: null }); }
        else post({ id: m.id, result: exec(v, m.job) });
      } catch (err) {
        post({ id: m.id, error: String(err) });
      }
    };
  }

  // k-way merge of sorted runs src[s..e) into out; before(a, b) = a goes first
  function kMerge(src, ranges, out, before) {
    const head = ranges.map((r) => r[0]), end = ranges.map((r) => r[1]), k = ranges.length;
    for (let o = 0; o < out.length; o++) {
      let best = -1;
      for (let r = 0; r < k; r++) {
        if (head[r] < end[r] && (best < 0 || before(src[head[r]], src[head[best]]))) best = r;
      }
      out[o] = src[head[best]++];
    }
    return out;
  }

  // -------------------------------------------------------------- the run
  async function run(n, pool) {
    const now = () => globalThis.performance.now();
    const tasks = [];
    const time = async (id, fn) => {
      const start = now();
      const result = await fn();
      tasks.push({ id, ms: now() - start, result });
    };

    const buf = allocate(n, pool.shared);
    const v = views(buf);
    await pool.init(buf);

    const k = Math.max(1, Math.min(pool.size, n));
    const ranges = [];
    for (let w = 0; w < k; w++) ranges.push([Math.floor((w * n) / k), Math.floor(((w + 1) * n) / k)]);
    const par = (op, arg) =>
      pool.run(ranges.map(([s, e], w) => ({ op, s, e, arg: typeof arg === "function" ? arg(w) : arg })));
    const custRange = (w) => ({ cs: Math.floor((w * N_CUST) / k), ce: Math.floor(((w + 1) * N_CUST) / k) });

    await time("generate", async () => { await par("gen", custRange); return null; });
    let qtySum = 0;
    for (let i = 0; i < n; i++) qtySum += v.qty[i];
    tasks[0].result = { rows: n, qty_sum: qtySum };

    await time("filter", async () => {
      const parts = await par("filter");
      return parts.reduce((a, p) => ({ count: a.count + p.count, revenue: a.revenue + p.revenue }), { count: 0, revenue: 0 });
    });

    await time("group_category", async () => {
      const parts = await par("groupCat");
      const sum = new Array(N_CAT).fill(0), cnt = new Array(N_CAT).fill(0);
      for (const p of parts) for (let c = 0; c < N_CAT; c++) { sum[c] += p.sum[c]; cnt[c] += p.cnt[c]; }
      let top = 0, best = -Infinity;
      for (let c = 0; c < N_CAT; c++) {
        if (cnt[c] > 0) { const m = sum[c] / cnt[c]; if (m > best) { best = m; top = c; } }
      }
      return { top_category: top, max_mean: best };
    });

    await time("group_region_category", async () => {
      const parts = await par("groupRegCat");
      const rev = new Array(N_REG * N_CAT).fill(0);
      for (const p of parts) for (let i = 0; i < rev.length; i++) rev[i] += p[i];
      let top = 0;
      for (let i = 1; i < rev.length; i++) if (rev[i] > rev[top]) top = i;
      return { top_key: top, top_revenue: rev[top] };
    });

    await time("join", async () => {
      const net = [0, 0, 0, 0];
      for (const p of await par("join")) for (let t = 0; t < 4; t++) net[t] += p[t];
      let top = 0;
      for (let t = 1; t < 4; t++) if (net[t] > net[top]) top = t;
      return { top_tier: top, top_net_revenue: net[top], net_revenue: net[0] + net[1] + net[2] + net[3] };
    });

    await time("group_customer", async () => {
      const rev = new Map();
      for (const p of await par("groupCust")) for (const [key, x] of p) rev.set(key, (rev.get(key) || 0) + x);
      let topKey = -1, best = -Infinity;
      for (const [key, x] of rev) if (x > best) { best = x; topKey = key; }
      return { customers: rev.size, top_customer: topKey, top_revenue: best };
    });

    await time("text", async () => {
      let matches = 0;
      const sums = new Map();
      for (const p of await par("text")) {
        matches += p.count;
        for (const [code, x] of p.sums) sums.set(code, (sums.get(code) || 0) + x);
      }
      let top = "", best = -Infinity;
      for (const [code, x] of sums) if (x > best) { best = x; top = code; }
      return { matches, top_sku: top, top_sku_value: best };
    });

    await time("sort", async () => {
      await par("sortIdx");                       // each worker sorts its range
      const val = v.val;
      const order = k > 1 ? kMerge(v.idx, ranges, new Uint32Array(n), (a, b) => val[a] > val[b]) : v.idx;
      let top10 = 0;
      for (let i = 0; i < Math.min(10, n); i++) top10 += val[order[i]];
      return { top_id: order[0], top10_sum: top10 };
    });

    await time("stats", async () => {
      const mean = (await par("sum")).reduce((a, b) => a + b, 0) / n;
      const sq = (await par("sqdev", mean)).reduce((a, b) => a + b, 0);
      await par("sortVals");
      const sorted = k > 1 ? kMerge(v.srt, ranges, new Float64Array(n), (a, b) => a < b) : v.srt;
      const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[(n - 1) / 2];
      const p95 = sorted[Math.floor((n - 1) * 0.95)];
      return { mean, std: Math.sqrt(sq / n), median, p95 };
    });

    const total = tasks.reduce((acc, t) => acc + t.ms, 0);
    return { rows: n, cores: k, tasks, total_ms: total };
  }

  return { run, createPool, localPool, workerHandler, exec, views, allocate };
});
