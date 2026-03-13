from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.runs import router as runs_router
from src.api.routes.ai import router as ai_router

app = FastAPI(title="Space Debris Engine API", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# /simulate/*
app.include_router(runs_router, prefix="/simulate", tags=["simulate"])

# /ai/*
app.include_router(ai_router, prefix="/ai", tags=["ai"])