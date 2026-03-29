const STT_URL = process.env.NEXT_PUBLIC_STT_URL || "http://localhost:8001";
const LLM_URL = process.env.NEXT_PUBLIC_LLM_URL || "http://localhost:8002";
const TTS_URL = process.env.NEXT_PUBLIC_TTS_URL || "http://localhost:8003";

export interface TranscribeResult {
  text: string;
  language: string;
  language_probability: number;
  duration: number;
  model: string;
}

export interface TranslateResult {
  translation: string;
  model: string;
  prompt_version: string;
  target_lang: string;
  latency_ms: number;
}

export async function transcribe(
  file: File | Blob,
  model = "small",
  language = "fr"
): Promise<TranscribeResult> {
  const form = new FormData();
  form.append("file", file, "audio.mp3");
  form.append("model", model);
  form.append("language", language);

  const res = await fetch(`${STT_URL}/transcribe`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `STT error ${res.status}`);
  }
  return res.json();
}

export async function translate(
  text: string,
  targetLang = "en",
  model = "groq/llama-3.1-8b-instant",
  promptVersion = "v1.1"
): Promise<TranslateResult> {
  const res = await fetch(`${LLM_URL}/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, target_lang: targetLang, model, prompt_version: promptVersion }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `LLM error ${res.status}`);
  }
  return res.json();
}

export async function synthesize(text: string, lang = "en"): Promise<Blob> {
  const res = await fetch(`${TTS_URL}/synthesize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, lang }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `TTS error ${res.status}`);
  }
  return res.blob();
}
