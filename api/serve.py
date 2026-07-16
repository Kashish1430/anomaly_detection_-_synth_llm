from __future__ import annotations

import asyncio
import sys

import uvicorn


def main() -> None:
    """Local dev entrypoint - prefer this over the bare `uvicorn api.main:app`
    CLI on Windows.

    uvicorn's own loop factory (uvicorn/loops/asyncio.py) hardcodes
    `asyncio.ProactorEventLoop` on Windows whenever it isn't running in
    subprocess-worker mode, regardless of `--loop` or any event loop policy
    set beforehand - so `asyncio.set_event_loop_policy(...)` in api/main.py
    has no effect on it. psycopg3's async pool cannot run on ProactorEventLoop
    at all: every connection attempt fails silently in the background
    (psycopg_pool retries and only logs a warning), surfacing later as a
    30-second pool-timeout 500 on the first request that needs Postgres.
    Driving `Server.serve()` ourselves inside `asyncio.run(...,
    loop_factory=SelectorEventLoop)` sidesteps uvicorn's hardcoded choice.

    Not needed once api/ runs in Docker (infra/Dockerfile.api, PLAN.md §09) -
    Linux has no Proactor/Selector split, so `uvicorn api.main:app` there
    works as-is.
    """
    config = uvicorn.Config("api.main:app", host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)

    if sys.platform == "win32":
        asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)
    else:
        asyncio.run(server.serve())


if __name__ == "__main__":
    main()
