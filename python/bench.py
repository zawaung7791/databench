"""Python benchmark engines: pure Python (multi-process), pandas, and Polars.

All build exactly the same dataset as the Rust and JavaScript versions
(mulberry32 PRNG streams) and compute the same results.
"""
import platform
import time

import numpy as np
import pandas as pd
import polars as pl

import pure_parallel

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


def _timed(tasks, task_id, fn):
    start = time.perf_counter()
    result = fn()
    tasks.append({"id": task_id, "ms": (time.perf_counter() - start) * 1000, "result": result})
    return result


def _run_tasks(tasks, fns):
    for task_id, fn in fns:
        _timed(tasks, task_id, fn)


# ------------------------------------------------------- vectorised generator
def _np_rand(seed, j):
    """mulberry32 outputs for counters j (state = seed + j*K), vectorised in NumPy."""
    s = ((seed + j * K) & M).astype(np.uint32)
    t = (s ^ (s >> 15)) * (s | 1)                     # uint32 math wraps like the others
    t = (t + (t ^ (t >> 7)) * (t | 61)) ^ t
    return (t ^ (t >> 14)).astype(np.float64) / 4294967296.0


def _np_columns(n):
    r = _np_rand(SEED, np.arange(1, 4 * n + 1, dtype=np.uint64)).reshape(n, 4)
    i1 = np.arange(1, n + 1, dtype=np.uint64)
    c1 = np.arange(1, N_CUST + 1, dtype=np.uint64)
    rows = {
        "category": (r[:, 0] * N_CAT).astype(np.int64),
        "region": (r[:, 1] * N_REG).astype(np.int64),
        "value": r[:, 2] * 1000.0,
        "qty": 1 + (r[:, 3] * 10).astype(np.int64),
        "cust_id": (_np_rand(SEED_CUST, i1) * N_CUST).astype(np.int64) * 7919 + 13,
        "pid": (_np_rand(SEED_PROD, i1) * N_PROD).astype(np.int64),
    }
    tier = (_np_rand(SEED_TIER, c1) * 4).astype(np.int64)
    customers = {
        "cust_id": np.arange(N_CUST, dtype=np.int64) * 7919 + 13,
        "tier": tier,
        "discount": np.array(DISC)[tier],
    }
    return rows, customers


# ------------------------------------------------------------ pandas / NumPy
def run_pandas(n):
    tasks = []
    holder = {}

    def generate():
        rows, customers = _np_columns(n)
        pid = pd.Series(rows.pop("pid"))
        df = pd.DataFrame(rows)
        df["sku"] = "SKU-" + pid.astype("str").str.zfill(4)
        holder["df"], holder["cust"] = df, pd.DataFrame(customers)
        return None

    _timed(tasks, "generate", generate)
    df, cust = holder["df"], holder["cust"]
    tasks[0]["result"] = {"rows": n, "qty_sum": int(df["qty"].sum())}

    def filt():
        sub = df[(df["value"] > 500.0) & (df["qty"] >= 5)]
        return {"count": int(len(sub)), "revenue": float((sub["value"] * sub["qty"]).sum())}

    def group_category():
        means = df.groupby("category")["value"].mean()
        return {"top_category": int(means.idxmax()), "max_mean": float(means.max())}

    def group_region_category():
        rev = (df["value"] * df["qty"]).groupby([df["region"], df["category"]]).sum()
        r, c = rev.idxmax()
        return {"top_key": int(r * N_CAT + c), "top_revenue": float(rev.max())}

    def join():
        m = df[["cust_id", "value", "qty"]].merge(cust, on="cust_id", how="inner")
        net = (m["value"] * m["qty"] * (1 - m["discount"])).groupby(m["tier"]).sum()
        top = int(net.idxmax())
        return {"top_tier": top, "top_net_revenue": float(net[top]), "net_revenue": float(net.sum())}

    def group_customer():
        rev = (df["value"] * df["qty"]).groupby(df["cust_id"]).sum()
        top = rev.idxmax()
        return {"customers": int(len(rev)), "top_customer": int(top), "top_revenue": float(rev[top])}

    def text():
        mask = df["sku"].str.contains("42", regex=False)
        sums = df.loc[mask, "value"].groupby(df.loc[mask, "sku"]).sum()
        top = sums.idxmax()
        return {"matches": int(mask.sum()), "top_sku": str(top), "top_sku_value": float(sums[top])}

    def sort():
        top = df.sort_values("value", ascending=False).head(10)
        return {"top_id": int(top.index[0]), "top10_sum": float(top["value"].sum())}

    def stats():
        v = df["value"].to_numpy()
        srt = np.sort(v)
        median = (srt[n // 2 - 1] + srt[n // 2]) / 2 if n % 2 == 0 else srt[n // 2]
        return {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "median": float(median),
            "p95": float(srt[int((n - 1) * 0.95)]),
        }

    _run_tasks(tasks, [("filter", filt), ("group_category", group_category),
                       ("group_region_category", group_region_category),
                       ("join", join), ("group_customer", group_customer), ("text", text),
                       ("sort", sort), ("stats", stats)])
    return {
        "rows": n,
        "cores": 1,  # pandas has no built-in multi-core execution
        "tasks": tasks,
        "total_ms": sum(t["ms"] for t in tasks),
        "language": "Python",
        "runtime": f"CPython {platform.python_version()}, pandas {pd.__version__}",
    }


# ------------------------------------------------------------------- Polars
def _pl_rand(lf, specs):
    """Add mulberry32 columns natively in Polars. specs: [(name, seed, counter_expr)].

    Math is done in UInt64 with explicit 32-bit masks, and right shifts are written as
    floor division, so no step can overflow. Intermediate columns keep each step computed once.
    """
    def u(x):
        return pl.lit(x, dtype=pl.UInt64)

    mask = u(M)
    lf = lf.with_columns([((u(seed) + ctr * u(K)) & mask).alias(f"_s_{nm}") for nm, seed, ctr in specs])
    lf = lf.with_columns([
        (((pl.col(f"_s_{nm}") ^ (pl.col(f"_s_{nm}") // u(1 << 15))) * (pl.col(f"_s_{nm}") | u(1))) & mask)
        .alias(f"_t_{nm}") for nm, _, _ in specs])
    lf = lf.with_columns([
        (((pl.col(f"_t_{nm}")
           + (((pl.col(f"_t_{nm}") ^ (pl.col(f"_t_{nm}") // u(1 << 7))) * (pl.col(f"_t_{nm}") | u(61))) & mask))
          & mask) ^ pl.col(f"_t_{nm}"))
        .alias(f"_w_{nm}") for nm, _, _ in specs])
    return lf.with_columns([
        ((pl.col(f"_w_{nm}") ^ (pl.col(f"_w_{nm}") // u(1 << 14))).cast(pl.Float64) / 4294967296.0).alias(nm)
        for nm, _, _ in specs])


def _gen_polars(n):
    """Generate everything natively in Polars (Arrow memory, Polars expression engine)."""
    def u(x):
        return pl.lit(x, dtype=pl.UInt64)

    i = pl.col("i")
    rows = _pl_rand(
        pl.LazyFrame().select(pl.int_range(0, n, dtype=pl.UInt64).alias("i")),
        [("r0", SEED, i * u(4) + u(1)), ("r1", SEED, i * u(4) + u(2)),
         ("r2", SEED, i * u(4) + u(3)), ("r3", SEED, i * u(4) + u(4)),
         ("rc", SEED_CUST, i + u(1)), ("rp", SEED_PROD, i + u(1))],
    ).select(
        pl.col("i").cast(pl.Int64).alias("id"),                  # row id, so the sort can report it
        (pl.col("r0") * N_CAT).cast(pl.Int64).alias("category"),
        (pl.col("r1") * N_REG).cast(pl.Int64).alias("region"),
        (pl.col("r2") * 1000.0).alias("value"),
        (1 + (pl.col("r3") * 10).cast(pl.Int64)).alias("qty"),
        ((pl.col("rc") * N_CUST).cast(pl.Int64) * 7919 + 13).alias("cust_id"),
        pl.concat_str([pl.lit("SKU-"),
                       (pl.col("rp") * N_PROD).cast(pl.Int64).cast(pl.String).str.pad_start(4, "0")])
        .alias("sku"),
    )
    tier = pl.col("tier")
    customers = _pl_rand(
        pl.LazyFrame().select(pl.int_range(0, N_CUST, dtype=pl.UInt64).alias("c")),
        [("rt", SEED_TIER, pl.col("c") + u(1))],
    ).select(
        (pl.col("c").cast(pl.Int64) * 7919 + 13).alias("cust_id"),
        (pl.col("rt") * 4).cast(pl.Int64).alias("tier"),
    ).with_columns(
        pl.when(tier == 0).then(pl.lit(DISC[0])).when(tier == 1).then(pl.lit(DISC[1]))
        .when(tier == 2).then(pl.lit(DISC[2])).otherwise(pl.lit(DISC[3])).alias("discount")
    )
    return pl.collect_all([rows, customers])


def run_polars(n):
    tasks = []
    holder = {}

    def generate():
        holder["df"], holder["cust"] = _gen_polars(n)
        return None

    _timed(tasks, "generate", generate)
    df, cust = holder["df"], holder["cust"]
    tasks[0]["result"] = {"rows": n, "qty_sum": int(df["qty"].sum())}

    def filt():
        sub = df.filter((pl.col("value") > 500.0) & (pl.col("qty") >= 5))
        revenue = sub.select((pl.col("value") * pl.col("qty")).sum()).item()
        return {"count": sub.height, "revenue": float(revenue)}

    def group_category():
        top = (df.group_by("category")
                 .agg(pl.col("value").mean().alias("mean"))
                 .sort("mean", descending=True)
                 .row(0, named=True))
        return {"top_category": int(top["category"]), "max_mean": float(top["mean"])}

    def group_region_category():
        top = (df.group_by(["region", "category"])
                 .agg((pl.col("value") * pl.col("qty")).sum().alias("rev"))
                 .sort("rev", descending=True)
                 .row(0, named=True))
        return {"top_key": int(top["region"] * N_CAT + top["category"]), "top_revenue": float(top["rev"])}

    def join():
        net = (df.select("cust_id", "value", "qty")
                 .join(cust, on="cust_id", how="inner")
                 .group_by("tier")
                 .agg((pl.col("value") * pl.col("qty") * (1 - pl.col("discount"))).sum().alias("net")))
        top = net.row(net["net"].arg_max(), named=True)
        return {"top_tier": int(top["tier"]), "top_net_revenue": float(top["net"]),
                "net_revenue": float(net["net"].sum())}

    def group_customer():
        rev = df.group_by("cust_id").agg((pl.col("value") * pl.col("qty")).sum().alias("rev"))
        top = rev.row(rev["rev"].arg_max(), named=True)
        return {"customers": rev.height, "top_customer": int(top["cust_id"]), "top_revenue": float(top["rev"])}

    def text():
        sub = df.filter(pl.col("sku").str.contains("42", literal=True))
        sums = sub.group_by("sku").agg(pl.col("value").sum().alias("v"))
        top = sums.row(sums["v"].arg_max(), named=True)
        return {"matches": sub.height, "top_sku": top["sku"], "top_sku_value": float(top["v"])}

    def sort():
        top = df.sort("value", descending=True).head(10)
        return {"top_id": int(top["id"][0]), "top10_sum": float(top["value"].sum())}

    def stats():
        k = int((n - 1) * 0.95)
        r = df.select(
            pl.col("value").mean().alias("mean"),
            pl.col("value").std(ddof=0).alias("std"),
            pl.col("value").median().alias("median"),
            pl.col("value").sort().gather(k).first().alias("p95"),
        ).row(0, named=True)
        return {key: float(v) for key, v in r.items()}

    _run_tasks(tasks, [("filter", filt), ("group_category", group_category),
                       ("group_region_category", group_region_category),
                       ("join", join), ("group_customer", group_customer), ("text", text),
                       ("sort", sort), ("stats", stats)])
    return {
        "rows": n,
        "cores": pl.thread_pool_size(),
        "tasks": tasks,
        "total_ms": sum(t["ms"] for t in tasks),
        "language": "Python",
        "runtime": f"CPython {platform.python_version()}, Polars {pl.__version__}",
    }


ENGINES = {"pure": pure_parallel.run, "pandas": run_pandas, "polars": run_polars}
