"""
LLM Service — Port 8002
POST /translate : texte FR -> traduction EN/UK via LiteLLM (GROQ ou Ollama)
GET  /health    : statut + modèle actif
"""

import os
import time

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

litellm.set_verbose = False

PROMPTS = {
    "v1.0": (
        "Translate the following French text to {lang}. "
        "Output only the translation, nothing else.\n\n{text}"
    ),
    "v1.1": (
        "You are a professional translator specializing in traffic and road safety announcements. "
        "Translate the following French traffic bulletin to {lang}. "
        "Keep road names (A6, N7...) as-is. Output only the translation.\n\n{text}"
    ),
    "v1.2": (
        "You are a professional translator. "
        "Translate this French road traffic report to {lang}. "
        "Preserve road identifiers (A6, N7, D roads). "
        "Use clear, concise language suitable for a radio broadcast. "
        "Output only the translated text.\n\n{text}"
    ),
}

LANG_LABELS = {
    "en": "English",
    "uk": "Ukrainian",
    "es": "Spanish",
    "de": "German",
}


def call_llm(prompt: str, model: str, timeout: int = 60) -> tuple[str, float]:
    t0 = time.perf_counter()
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        timeout=timeout,
    )
    translation = response.choices[0].message.content.strip()
    latency_ms = (time.perf_counter() - t0) * 1000
    return translation, latency_ms

app = FastAPI(title="LLM Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
