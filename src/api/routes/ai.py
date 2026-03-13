from typing import Any, Dict, Optional
import os

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class AiRequest(BaseModel):
    prompt: str
    system: Optional[str] = None

def _provider() -> str:
    # "groq" | "gemini" | "disabled"
    return str(os.getenv("AI_PROVIDER", "disabled")).strip().lower()

def _groq_chat(prompt: str, system: str) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="AI is disabled: missing GROQ_API_KEY (set AI_PROVIDER=groq)")

    model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }

    r = requests.post(url, headers=headers, json=body, timeout=25)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Groq error: {r.status_code} {r.text[:500]}")

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail="Groq response parse failed")

def _gemini_generate(prompt: str, system: str) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="AI is disabled: missing GEMINI_API_KEY (set AI_PROVIDER=gemini)")

    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": system}],
        },
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
        },
    }

    r = requests.post(url, json=body, timeout=25)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini error: {r.status_code} {r.text[:500]}")

    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise HTTPException(status_code=502, detail="Gemini response parse failed")

@router.post("/help")
def help(req: AiRequest) -> Dict[str, Any]:
    prov = _provider()
    if prov == "disabled":
        raise HTTPException(
            status_code=503,
            detail="AI disabled. Set AI_PROVIDER=groq and GROQ_API_KEY, or AI_PROVIDER=gemini and GEMINI_API_KEY."
        )

    system = req.system or "You are a mission assistant for a space debris collision avoidance dashboard."
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt required")

    if prov == "groq":
        text = _groq_chat(prompt, system)
        return {"ok": True, "provider": "groq", "message": text}

    if prov == "gemini":
        text = _gemini_generate(prompt, system)
        return {"ok": True, "provider": "gemini", "message": text}

    raise HTTPException(status_code=400, detail=f"Unknown AI_PROVIDER={prov!r}")