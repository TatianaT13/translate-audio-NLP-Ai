"""
LLM Service — Port 8002
POST /translate : texte FR -> traduction EN/UK via LiteLLM (GROQ ou Ollama)
GET  /health    : statut + modèle actif
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.run_pipeline import PROMPTS, LANG_LABELS, call_llm

app = FastAPI(title="LLM Service", version="1.0.0")

DEFAULT_MODEL = os.getenv("LLM_MODEL", "groq/llama-3.1-8b-instant")
DEFAULT_PROMPT = os.getenv("PROMPT_VERSION", "v1.1")


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "en"
    model: str = DEFAULT_MODEL
    prompt_version: str = DEFAULT_PROMPT


class TranslateResponse(BaseModel):
    translation: str
    model: str
    prompt_version: str
    target_lang: str
    latency_ms: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "default_prompt": DEFAULT_PROMPT,
        "available_prompts": list(PROMPTS.keys()),
    }


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    """
    Traduit un texte via LiteLLM.

    - **text** : texte source (français)
    - **target_lang** : langue cible (en, uk...)
    - **model** : modèle LiteLLM (groq/llama-3.1-8b-instant, ollama/mistral...)
    - **prompt_version** : version du prompt (v1.0, v1.1, v1.2)
    """
    if req.prompt_version not in PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"prompt_version invalide. Valeurs acceptées : {list(PROMPTS.keys())}",
        )
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Le champ text est vide.")

    lang_label = LANG_LABELS.get(req.target_lang, req.target_lang)
    prompt = PROMPTS[req.prompt_version].format(lang=lang_label, text=req.text)

    try:
        translation, latency_ms = call_llm(prompt=prompt, model=req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return TranslateResponse(
        translation=translation,
        model=req.model,
        prompt_version=req.prompt_version,
        target_lang=req.target_lang,
        latency_ms=int(latency_ms),
    )
