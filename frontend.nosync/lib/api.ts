const PIPELINE_URL = process.env.NEXT_PUBLIC_PIPELINE_URL || "http://localhost:8000";

export interface ProcessResult {
  source_text: string;
  language: string;
  language_prob: number;
  translation: string;
  audio_b64: string;
  audio_content_type: string;
  latency_stt_ms: number;
  latency_llm_ms: number;
  latency_tts_ms: number;
  latency_total_ms: number;
}

export async function process(
  file: File | Blob,
  targetLang = "en",
  llmModel = "groq/llama-3.1-8b-instant",
  promptVersion = "v1.1",
  whisperModel = "small"
): Promise<ProcessResult> {
  const form = new FormData();
  form.append("file", file, "audio.mp3");
  form.append("target_lang", targetLang);
  form.append("llm_model", llmModel);
  form.append("prompt_version", promptVersion);
  form.append("whisper_model", whisperModel);

  const res = await fetch(`${PIPELINE_URL}/process`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Pipeline error ${res.status}`);
  }
  return res.json();
}

export function audioFromBase64(b64: string, contentType: string): Blob {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: contentType });
}
