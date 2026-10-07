import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

import pure_parallel
from bench import ENGINES

MAX_ROWS = int(os.environ.get("MAX_ROWS", "2000000"))
lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app):
    pure_parallel.get_pool()          # start worker processes before the first run
    yield
    if pure_parallel._pool is not None:
        pure_parallel._pool.terminate()


app = FastAPI(title="Python data benchmark", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "workers": pure_parallel.WORKERS}


@app.get("/bench")
def bench(rows: int = 1_000_000, engine: str = "pure"):
    if engine not in ENGINES:
        raise HTTPException(400, f"Unknown engine '{engine}'. Use 'pure', 'pandas' or 'polars'.")
    n = max(1000, min(rows, MAX_ROWS))
    # One benchmark at a time, so runs don't fight each other for CPU.
    if not lock.acquire(blocking=False):
        raise HTTPException(429, "A Python benchmark is already running. Try again when it finishes.")
    try:
        return ENGINES[engine](n)
    finally:
        lock.release()
