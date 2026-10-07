# Data bench: Rust vs Python vs JavaScript

A small website that runs the same data-analytics workload in six engines and
compares their speed side by side:

| Engine | How it's built |
|---|---|
| Rust | axum server, release build, rayon across all cores |
| JavaScript | Node.js 22, typed arrays in shared memory, one worker thread per core |
| Python + pandas | FastAPI, NumPy/pandas vectorised, one core (pandas has no built-in parallelism) |
| Python + Polars | FastAPI, fully Polars-native including data generation, all cores |
| Python (pure) | FastAPI, plain loops, one process per core over shared memory |
| JavaScript in browser | the same `bench.js` in one Web Worker per core on the visitor's device |

Every engine generates the identical dataset (mulberry32 PRNG streams), then runs:
generate, filter, group by category, group by region + category, join against a
100,000-row customer table, group by customer (up to 100,000 groups), text search
and group-by on a string column, full sort, and summary statistics. The page cross-checks every result, so a wrong implementation
shows up as a mismatch instead of a suspiciously fast time.

## Layout

```
docker-compose.yml
web/      nginx: static page + reverse proxy to /api/{rust,python,node}/
rust/     axum service      (GET /bench?rows=N)
python/   FastAPI service   (GET /bench?rows=N&engine=pure|pandas|polars)
node/     plain http service (GET /bench?rows=N), bench.js is shared with the browser
```

## Run it on the VPS

```bash
git clone https://github.com/<you>/databench.git && cd databench
docker compose up -d --build        # first build compiles Rust, takes a few minutes
curl -s 127.0.0.1:8090/api/rust/bench?rows=100000 | head -c 300
```

The site listens on `127.0.0.1:8090` only. Change the port in `docker-compose.yml`
if 8090 is taken.

### Subdomain: bench.zawaung.site

Point a DNS A record for `bench` at the VPS, then add one of these.

**nginx**

```nginx
server {
    server_name bench.zawaung.site;
    location / {
        proxy_pass http://127.0.0.1:8090;
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
    }
}
```
then `sudo nginx -t && sudo systemctl reload nginx && sudo certbot --nginx -d bench.zawaung.site`

**Caddy**

```
bench.zawaung.site {
    reverse_proxy 127.0.0.1:8090
}
```

## Tuning

- `MAX_ROWS` (default 2,000,000) caps the request size per engine; raise it and
  `mem_limit` together if your VPS has the RAM.
- Every engine uses all cores by default. To compare at a fixed core count, set
  `RAYON_NUM_THREADS` (Rust), `BENCH_WORKERS` (Node and pure Python) and
  `POLARS_MAX_THREADS` (Polars) in `docker-compose.yml`.
- The browser lane needs the page to be cross-origin isolated (the COOP/COEP
  headers in `web/nginx.conf`); without them it falls back to one core.
- Each service runs one benchmark at a time and answers 429 if busy, so a public
  page can't be used to pile up work on the server.

## Update / stop

```bash
git pull && docker compose up -d --build
docker compose down
```
