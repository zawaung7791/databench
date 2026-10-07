"""Pure-Python engine on all cores.

The GIL stops threads from running Python code in parallel, so this uses one process
per core. Numeric columns live in shared memory; each process always owns the same
range of rows (and keeps that range's product-code strings itself), and the parent
combines partial results (merging sorted runs with heapq.merge).
Standard library only, so worker processes start quickly.
"""
import heapq
import os
import platform
import time
from array import array
from multiprocessing import get_context, shared_memory

SEED = 42          # main stream: category, region, value, quantity
SEED_CUST = 1001   # which customer each row belongs to
SEED_PROD = 2002   # which product code each row has
SEED_TIER = 3003   # each customer's tier
K = 0x6D2B79F5
M = 0xFFFFFFFF
N_CAT = 20
N_REG = 5
N_CUST = 100_000
N_PROD = 5000
DISC = [0.0, 0.05, 0.1, 0.2]   # discount per customer tier
COLUMNS = {"cat": "B", "reg": "B", "val": "d", "qty": "I", "idx": "I", "srt": "d", "cref": "I"}
CUST_COLUMNS = {"cid": "I", "ctier": "B"}
FORMATS = {**COLUMNS, **CUST_COLUMNS}
ITEM = {"B": 1, "d": 8, "I": 4}

WORKERS = int(os.environ.get("BENCH_WORKERS", "0")) or os.cpu_count() or 1
_pool = None


def cust_id(c):
    return c * 7919 + 13   # sparse ids, so a real hash lookup is needed


class Workers:
    """One process per core; job w always goes to process w, so each process can
    keep state (its range's strings) between jobs."""

    def __init__(self, k):
        ctx = get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(k):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker_loop, args=(child,), daemon=True)
            proc.start()
            self.conns.append(parent)
            self.procs.append(proc)

    def map(self, msgs):
        for conn, msg in zip(self.conns, msgs):
            conn.send(msg)
        out = [conn.recv() for conn in self.conns[:len(msgs)]]
        for status, value in out:
            if status != "ok":
                raise RuntimeError(value)
        return [value for _, value in out]

    def terminate(self):
        for conn in self.conns:
            conn.send(None)
        for proc in self.procs:
            proc.join(timeout=2)


def get_pool():
    """Start the worker processes once, so start-up isn't counted in the timings."""
    global _pool
    if _pool is None:
        _pool = Workers(WORKERS)
    return _pool


def _mulberry(s):
    """mulberry32 output for generator state s (= SEED + j*K)."""
    t = ((s ^ (s >> 15)) * (s | 1)) & M
    t = ((t + (((t ^ (t >> 7)) * (t | 61)) & M)) & M) ^ t
    return (t ^ (t >> 14)) / 4294967296.0


# --------------------------------------------------------------- shared memory
def _attach(names):
    shms = {}
    for key, name in names.items():
        try:
            shms[key] = shared_memory.SharedMemory(name=name, track=False)  # Python 3.13+
        except TypeError:
            shms[key] = shared_memory.SharedMemory(name=name)
    return shms, {key: shm.buf.cast(FORMATS[key]) for key, shm in shms.items()}


def _detach(shms, views):
    for v in views.values():
        v.release()
    for shm in shms.values():
        shm.close()


# ------------------------------------------------------ kernels (one range each)
def _gen(v, s, e, arg, local):
    cat, reg, val, qty, cref = v["cat"], v["reg"], v["val"], v["qty"], v["cref"]
    sku = [""] * (e - s)
    for i in range(s, e):
        j = 4 * i
        cat[i] = int(_mulberry((SEED + (j + 1) * K) & M) * N_CAT)
        reg[i] = int(_mulberry((SEED + (j + 2) * K) & M) * N_REG)
        val[i] = _mulberry((SEED + (j + 3) * K) & M) * 1000.0
        qty[i] = 1 + int(_mulberry((SEED + (j + 4) * K) & M) * 10)
        cref[i] = cust_id(int(_mulberry((SEED_CUST + (i + 1) * K) & M) * N_CUST))
        sku[i - s] = f"SKU-{int(_mulberry((SEED_PROD + (i + 1) * K) & M) * N_PROD):04d}"
    local["sku"] = sku
    cs, ce = arg                       # this worker's share of the customer table
    cid, ctier = v["cid"], v["ctier"]
    for c in range(cs, ce):
        cid[c] = cust_id(c)
        ctier[c] = int(_mulberry((SEED_TIER + (c + 1) * K) & M) * 4)
    return None


def _join(v, s, e, _, local):
    tier_of = dict(zip(v["cid"][:N_CUST].tolist(), v["ctier"][:N_CUST].tolist()))   # hash index on customers
    net = [0.0, 0.0, 0.0, 0.0]
    for r, x, q in zip(v["cref"][s:e].tolist(), v["val"][s:e].tolist(), v["qty"][s:e].tolist()):
        t = tier_of[r]
        net[t] += x * q * (1 - DISC[t])
    return net


def _group_cust(v, s, e, _, local):
    rev = {}
    for r, x, q in zip(v["cref"][s:e].tolist(), v["val"][s:e].tolist(), v["qty"][s:e].tolist()):
        rev[r] = rev.get(r, 0.0) + x * q
    return rev


def _text(v, s, e, _, local):
    count = 0
    sums = {}
    for code, x in zip(local["sku"], v["val"][s:e].tolist()):
        if "42" in code:
            count += 1
            sums[code] = sums.get(code, 0.0) + x
    return count, sums


def _filter(v, s, e, _, local):
    count, revenue = 0, 0.0
    for x, q in zip(v["val"][s:e].tolist(), v["qty"][s:e].tolist()):
        if x > 500.0 and q >= 5:
            count += 1
            revenue += x * q
    return count, revenue


def _group_cat(v, s, e, _, local):
    sums = [0.0] * N_CAT
    cnts = [0] * N_CAT
    for c, x in zip(v["cat"][s:e].tolist(), v["val"][s:e].tolist()):
        sums[c] += x
        cnts[c] += 1
    return sums, cnts


def _group_reg_cat(v, s, e, _, local):
    rev = [0.0] * (N_REG * N_CAT)
    for r, c, x, q in zip(v["reg"][s:e].tolist(), v["cat"][s:e].tolist(),
                          v["val"][s:e].tolist(), v["qty"][s:e].tolist()):
        rev[r * N_CAT + c] += x * q
    return rev


def _sort_idx(v, s, e, _, local):
    vals = v["val"][s:e].tolist()
    order = sorted(range(e - s), key=vals.__getitem__, reverse=True)
    v["idx"][s:e] = array("I", [s + o for o in order])
    return None


def _sum(v, s, e, _, local):
    t = 0.0
    for x in v["val"][s:e].tolist():
        t += x
    return t


def _sqdev(v, s, e, mean, local):
    t = 0.0
    for x in v["val"][s:e].tolist():
        d = x - mean
        t += d * d
    return t


def _sort_vals(v, s, e, _, local):
    v["srt"][s:e] = array("d", sorted(v["val"][s:e].tolist()))
    return None


OPS = {"gen": _gen, "filter": _filter, "group_cat": _group_cat, "group_reg_cat": _group_reg_cat,
       "sort_idx": _sort_idx, "sum": _sum, "sqdev": _sqdev, "sort_vals": _sort_vals,
       "join": _join, "group_cust": _group_cust, "text": _text}


def _worker_loop(conn):
    local = {}
    while True:
        msg = conn.recv()
        if msg is None:
            break
        op, names, s, e, arg = msg
        try:
            shms, views = _attach(names)
            try:
                result = OPS[op](views, s, e, arg, local)
            finally:
                _detach(shms, views)
            conn.send(("ok", result))
        except Exception as exc:  # report back instead of hanging the parent
            conn.send(("error", f"{op}: {exc!r}"))


# ------------------------------------------------------------------- the run
def run(n):
    pool = get_pool()
    k = max(1, min(WORKERS, n))
    ranges = [(w * n // k, (w + 1) * n // k) for w in range(k)]
    sizes = {**{key: n for key in COLUMNS}, **{key: N_CUST for key in CUST_COLUMNS}}
    shms = {key: shared_memory.SharedMemory(create=True, size=max(1, sizes[key] * ITEM[fmt]))
            for key, fmt in FORMATS.items()}
    names = {key: shm.name for key, shm in shms.items()}
    views = {key: shm.buf.cast(FORMATS[key]) for key, shm in shms.items()}
    tasks = []

    def par(op, arg=None):
        args = arg if callable(arg) else (lambda w: arg)
        return pool.map([(op, names, s, e, args(w)) for w, (s, e) in enumerate(ranges)])

    def cust_range(w):
        return (w * N_CUST // k, (w + 1) * N_CUST // k)

    def timed(task_id, fn):
        start = time.perf_counter()
        result = fn()
        tasks.append({"id": task_id, "ms": (time.perf_counter() - start) * 1000, "result": result})

    def filt():
        parts = par("filter")
        return {"count": sum(p[0] for p in parts), "revenue": sum(p[1] for p in parts)}

    def group_category():
        sums = [0.0] * N_CAT
        cnts = [0] * N_CAT
        for ps, pc in par("group_cat"):
            for c in range(N_CAT):
                sums[c] += ps[c]
                cnts[c] += pc[c]
        top, best = 0, float("-inf")
        for c in range(N_CAT):
            if cnts[c] and sums[c] / cnts[c] > best:
                best, top = sums[c] / cnts[c], c
        return {"top_category": top, "max_mean": best}

    def group_region_category():
        rev = [0.0] * (N_REG * N_CAT)
        for part in par("group_reg_cat"):
            for i, x in enumerate(part):
                rev[i] += x
        top = max(range(len(rev)), key=rev.__getitem__)
        return {"top_key": top, "top_revenue": rev[top]}

    def join():
        net = [0.0, 0.0, 0.0, 0.0]
        for part in par("join"):
            for t in range(4):
                net[t] += part[t]
        top = max(range(4), key=net.__getitem__)
        return {"top_tier": top, "top_net_revenue": net[top], "net_revenue": net[0] + net[1] + net[2] + net[3]}

    def group_customer():
        rev = {}
        for part in par("group_cust"):
            for key, x in part.items():
                rev[key] = rev.get(key, 0.0) + x
        top = max(rev, key=rev.__getitem__)
        return {"customers": len(rev), "top_customer": top, "top_revenue": rev[top]}

    def text():
        matches = 0
        sums = {}
        for count, part in par("text"):
            matches += count
            for code, x in part.items():
                sums[code] = sums.get(code, 0.0) + x
        top = max(sums, key=sums.__getitem__)
        return {"matches": matches, "top_sku": top, "top_sku_value": sums[top]}

    def sort():
        par("sort_idx")
        idx = views["idx"][:n].tolist()
        vals = views["val"][:n].tolist()
        order = list(heapq.merge(*(idx[s:e] for s, e in ranges), key=vals.__getitem__, reverse=True))
        return {"top_id": order[0], "top10_sum": sum(vals[i] for i in order[:10])}

    def stats():
        mean = sum(par("sum")) / n
        var = sum(par("sqdev", mean)) / n
        par("sort_vals")
        srt = views["srt"][:n].tolist()
        merged = list(heapq.merge(*(srt[s:e] for s, e in ranges)))
        median = (merged[n // 2 - 1] + merged[n // 2]) / 2 if n % 2 == 0 else merged[n // 2]
        return {"mean": mean, "std": var ** 0.5, "median": median, "p95": merged[int((n - 1) * 0.95)]}

    try:
        timed("generate", lambda: par("gen", cust_range))
        tasks[0]["result"] = {"rows": n, "qty_sum": sum(views["qty"][:n].tolist())}
        for task_id, fn in [("filter", filt), ("group_category", group_category),
                            ("group_region_category", group_region_category),
                            ("join", join), ("group_customer", group_customer), ("text", text),
                            ("sort", sort), ("stats", stats)]:
            timed(task_id, fn)
    finally:
        for v in views.values():
            v.release()
        for shm in shms.values():
            shm.close()
            shm.unlink()

    return {
        "rows": n,
        "cores": k,
        "tasks": tasks,
        "total_ms": sum(t["ms"] for t in tasks),
        "language": "Python",
        "runtime": f"CPython {platform.python_version()}, multiprocessing",
    }
