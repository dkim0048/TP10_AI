import time
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import image_classification, word_difficulty

RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW = 60  # seconds

_rate_limit_store: dict[str, list[float]] = defaultdict(list)

app = FastAPI(title="Lexical Bridge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dysletp10.vercel.app"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    timestamps = _rate_limit_store[ip]
    timestamps[:] = [t for t in timestamps if t > window_start]
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please try again later."},
        )
    timestamps.append(now)
    return await call_next(request)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


app.include_router(word_difficulty.router)
app.include_router(image_classification.router)
