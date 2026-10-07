//! Rust benchmark on all cores (rayon): same dataset (mulberry32, seed 42) and
//! same tasks as the Python and JavaScript versions.
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{sync::Arc, time::Instant};
use tokio::sync::Mutex;

const SEED: u32 = 42; // main stream: category, region, value, quantity
const SEED_CUST: u32 = 1001; // which customer each row belongs to
const SEED_PROD: u32 = 2002; // which product code each row has
const SEED_TIER: u32 = 3003; // each customer's tier
const N_CAT: usize = 20;
const N_REG: usize = 5;
const N_CUST: usize = 100_000;
const N_PROD: f64 = 5000.0;
const DISC: [f64; 4] = [0.0, 0.05, 0.1, 0.2]; // discount per customer tier

/// Sparse customer ids, so a real hash lookup is needed
fn cust_id(c: u32) -> u32 {
    c * 7919 + 13
}

/// mulberry32 output for counter j (the generator state is just SEED + j*K),
/// so every row can be generated independently, on any core.
#[inline]
fn rand_with(seed: u32, j: u64) -> f64 {
    let a = seed.wrapping_add((j as u32).wrapping_mul(0x6D2B79F5));
    let mut t = (a ^ (a >> 15)).wrapping_mul(a | 1);
    t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61)) ^ t;
    (t ^ (t >> 14)) as f64 / 4294967296.0
}

#[inline]
fn rand(j: u64) -> f64 {
    rand_with(SEED, j)
}

fn timed<T>(tasks: &mut Vec<Value>, id: &str, f: impl FnOnce() -> (T, Value)) -> T {
    let start = Instant::now();
    let (keep, result) = f();
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    tasks.push(json!({ "id": id, "ms": ms, "result": result }));
    keep
}

fn run(n: usize) -> Value {
    let mut tasks = Vec::new();

    let (cat, reg, val, qty, cref, sku, cid, ctier) = timed(&mut tasks, "generate", || {
        let mut cat = vec![0u8; n];
        let mut reg = vec![0u8; n];
        let mut val = vec![0.0f64; n];
        let mut qty = vec![0u32; n];
        cat.par_iter_mut()
            .zip(reg.par_iter_mut())
            .zip(val.par_iter_mut())
            .zip(qty.par_iter_mut())
            .enumerate()
            .for_each(|(i, (((c, r), v), q))| {
                let j = 4 * i as u64;
                *c = (rand(j + 1) * N_CAT as f64) as u8;
                *r = (rand(j + 2) * N_REG as f64) as u8;
                *v = rand(j + 3) * 1000.0;
                *q = 1 + (rand(j + 4) * 10.0) as u32;
            });
        let cref: Vec<u32> = (0..n)
            .into_par_iter()
            .map(|i| cust_id((rand_with(SEED_CUST, i as u64 + 1) * N_CUST as f64) as u32))
            .collect();
        let sku: Vec<String> = (0..n)
            .into_par_iter()
            .map(|i| format!("SKU-{:04}", (rand_with(SEED_PROD, i as u64 + 1) * N_PROD) as u32))
            .collect();
        let (cid, ctier): (Vec<u32>, Vec<u8>) = (0..N_CUST as u32)
            .into_par_iter()
            .map(|c| (cust_id(c), (rand_with(SEED_TIER, c as u64 + 1) * 4.0) as u8))
            .unzip();
        ((cat, reg, val, qty, cref, sku, cid, ctier), Value::Null)
    });
    let qty_sum: u64 = qty.iter().map(|&q| q as u64).sum();
    tasks[0]["result"] = json!({ "rows": n, "qty_sum": qty_sum });

    timed(&mut tasks, "filter", || {
        let (count, revenue) = (0..n)
            .into_par_iter()
            .fold(
                || (0u64, 0.0f64),
                |(c, r), i| {
                    if val[i] > 500.0 && qty[i] >= 5 {
                        (c + 1, r + val[i] * qty[i] as f64)
                    } else {
                        (c, r)
                    }
                },
            )
            .reduce(|| (0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        ((), json!({ "count": count, "revenue": revenue }))
    });

    timed(&mut tasks, "group_category", || {
        let (sum, cnt) = (0..n)
            .into_par_iter()
            .fold(
                || ([0.0f64; N_CAT], [0u64; N_CAT]),
                |(mut s, mut c), i| {
                    let k = cat[i] as usize;
                    s[k] += val[i];
                    c[k] += 1;
                    (s, c)
                },
            )
            .reduce(
                || ([0.0f64; N_CAT], [0u64; N_CAT]),
                |(mut s1, mut c1), (s2, c2)| {
                    for k in 0..N_CAT {
                        s1[k] += s2[k];
                        c1[k] += c2[k];
                    }
                    (s1, c1)
                },
            );
        let (mut top, mut best) = (0usize, f64::NEG_INFINITY);
        for c in 0..N_CAT {
            if cnt[c] > 0 {
                let m = sum[c] / cnt[c] as f64;
                if m > best {
                    best = m;
                    top = c;
                }
            }
        }
        ((), json!({ "top_category": top, "max_mean": best }))
    });

    timed(&mut tasks, "group_region_category", || {
        let rev = (0..n)
            .into_par_iter()
            .fold(
                || [0.0f64; N_REG * N_CAT],
                |mut acc, i| {
                    acc[reg[i] as usize * N_CAT + cat[i] as usize] += val[i] * qty[i] as f64;
                    acc
                },
            )
            .reduce(
                || [0.0f64; N_REG * N_CAT],
                |mut a, b| {
                    for k in 0..a.len() {
                        a[k] += b[k];
                    }
                    a
                },
            );
        let mut top = 0usize;
        for k in 1..rev.len() {
            if rev[k] > rev[top] {
                top = k;
            }
        }
        ((), json!({ "top_key": top, "top_revenue": rev[top] }))
    });

    timed(&mut tasks, "join", || {
        // hash index on the customer table, then probe it from every core
        let tier_of: FxHashMap<u32, u8> = cid.iter().copied().zip(ctier.iter().copied()).collect();
        let net = (0..n)
            .into_par_iter()
            .fold(
                || [0.0f64; 4],
                |mut acc, i| {
                    let t = tier_of[&cref[i]] as usize;
                    acc[t] += val[i] * qty[i] as f64 * (1.0 - DISC[t]);
                    acc
                },
            )
            .reduce(
                || [0.0f64; 4],
                |mut a, b| {
                    for t in 0..4 {
                        a[t] += b[t];
                    }
                    a
                },
            );
        let mut top = 0usize;
        for t in 1..4 {
            if net[t] > net[top] {
                top = t;
            }
        }
        let total = net[0] + net[1] + net[2] + net[3];
        ((), json!({ "top_tier": top, "top_net_revenue": net[top], "net_revenue": total }))
    });

    timed(&mut tasks, "group_customer", || {
        let rev = (0..n)
            .into_par_iter()
            .fold(FxHashMap::<u32, f64>::default, |mut m, i| {
                *m.entry(cref[i]).or_insert(0.0) += val[i] * qty[i] as f64;
                m
            })
            .reduce(FxHashMap::<u32, f64>::default, |mut a, b| {
                for (k, x) in b {
                    *a.entry(k).or_insert(0.0) += x;
                }
                a
            });
        let (mut top, mut best) = (0u32, f64::NEG_INFINITY);
        for (&k, &x) in &rev {
            if x > best {
                best = x;
                top = k;
            }
        }
        ((), json!({ "customers": rev.len(), "top_customer": top, "top_revenue": best }))
    });

    timed(&mut tasks, "text", || {
        let (matches, sums) = (0..n)
            .into_par_iter()
            .fold(
                || (0u64, FxHashMap::<&str, f64>::default()),
                |(mut c, mut m), i| {
                    if sku[i].contains("42") {
                        c += 1;
                        *m.entry(sku[i].as_str()).or_insert(0.0) += val[i];
                    }
                    (c, m)
                },
            )
            .reduce(
                || (0u64, FxHashMap::<&str, f64>::default()),
                |(c1, mut a), (c2, b)| {
                    for (k, x) in b {
                        *a.entry(k).or_insert(0.0) += x;
                    }
                    (c1 + c2, a)
                },
            );
        let (mut top, mut best) = ("", f64::NEG_INFINITY);
        for (&k, &x) in &sums {
            if x > best {
                best = x;
                top = k;
            }
        }
        ((), json!({ "matches": matches, "top_sku": top, "top_sku_value": best }))
    });

    timed(&mut tasks, "sort", || {
        let mut idx: Vec<u32> = (0..n as u32).collect();
        idx.par_sort_unstable_by(|&a, &b| val[b as usize].total_cmp(&val[a as usize]));
        let top10: f64 = idx.iter().take(10).map(|&i| val[i as usize]).sum();
        ((), json!({ "top_id": idx[0], "top10_sum": top10 }))
    });

    timed(&mut tasks, "stats", || {
        let mean = val.par_iter().sum::<f64>() / n as f64;
        let var = val.par_iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        let mut sorted = val.clone();
        sorted.par_sort_unstable_by(|a, b| a.total_cmp(b));
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let p95 = sorted[((n - 1) as f64 * 0.95) as usize];
        ((), json!({ "mean": mean, "std": var.sqrt(), "median": median, "p95": p95 }))
    });

    let total: f64 = tasks.iter().map(|t| t["ms"].as_f64().unwrap_or(0.0)).sum();
    json!({
        "rows": n,
        "cores": rayon::current_num_threads(),
        "tasks": tasks,
        "total_ms": total,
        "language": "Rust",
        "runtime": option_env!("RUSTC_VERSION").unwrap_or("rustc"),
    })
}

#[derive(Deserialize)]
struct Params {
    rows: Option<usize>,
}

#[derive(Clone)]
struct AppState {
    lock: Arc<Mutex<()>>,
    max_rows: usize,
}

async fn bench(State(st): State<AppState>, Query(p): Query<Params>) -> Response {
    let n = p.rows.unwrap_or(1_000_000).clamp(1000, st.max_rows);
    // One benchmark at a time, so runs don't fight each other for CPU.
    let guard = match st.lock.clone().try_lock_owned() {
        Ok(g) => g,
        Err(_) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({ "error": "A Rust benchmark is already running. Try again when it finishes." })),
            )
                .into_response()
        }
    };
    match tokio::task::spawn_blocking(move || {
        let _held = guard;
        run(n)
    })
    .await
    {
        Ok(out) => Json(out).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

#[tokio::main]
async fn main() {
    let max_rows = std::env::var("MAX_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_000_000);
    let state = AppState { lock: Arc::new(Mutex::new(())), max_rows };
    let app = Router::new()
        .route("/bench", get(bench))
        .route("/health", get(|| async { Json(json!({ "ok": true })) }))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    println!("rust bench listening on 8000");
    axum::serve(listener, app).await.unwrap();
}
