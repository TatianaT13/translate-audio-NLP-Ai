import { getAccessToken, refreshAccessToken } from "@/lib/auth";

const PIPELINE_URL = process.env.NEXT_PUBLIC_PIPELINE_URL || "http://localhost:8000";
const LLM_URL      = process.env.NEXT_PUBLIC_LLM_URL      || "http://localhost:8002";
const STT_URL      = process.env.NEXT_PUBLIC_STT_URL      || "http://localhost:8001";

async function authHeaders(): Promise<HeadersInit> {
  let token = getAccessToken();
  if (!token) token = await refreshAccessToken();
  return token ? { "Authorization": `Bearer ${token}` } : {};
}

export interface ChunkTranscript {
  text:          string;
  language:      string;
  confidence:    number;
  duration_s:    number;
}

/** Envoie un chunk audio (Blob WebM/MP4) au STT pour transcription. */
export async function transcribeChunk(
  audio: Blob,
  whisperModel: string = "small"
): Promise<ChunkTranscript> {
  const form = new FormData();
  form.append("file", audio, "chunk.webm");
  form.append("model", whisperModel);

  const headers = await authHeaders();
  const res = await fetch(`${STT_URL}/transcribe`, {
    method: "POST",
    headers,
    body: form,
  });
  if (!res.ok) throw new Error(`STT erreur ${res.status}`);
  const data = await res.json();
  return {
    text:       data.text || "",
    language:   data.language || "",
    confidence: data.language_probability ?? 0,
    duration_s: data.duration ?? 0,
  };
}

export interface SummaryResponse {
  summary:           string;
  style:             string;
  target_lang:       string;
  model:             string;
  latency_ms:        number;
  prompt_tokens?:    number;
  completion_tokens?: number;
  total_tokens?:     number;
  cost_usd?:         number;
}

/** Génère un compte-rendu à partir d'un transcript complet. */
export async function summarizeMeeting(
  transcript: string,
  targetLang: string = "en",
  style: "executive" | "detailed" | "actions_only" = "executive"
): Promise<SummaryResponse> {
  const headers = await authHeaders();
  const res = await fetch(`${LLM_URL}/summarize`, {
    method: "POST",
    headers: { ...headers, "Content-Type": "application/json" },
    body: JSON.stringify({ transcript, target_lang: targetLang, style }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `LLM erreur ${res.status}`);
  }
  return res.json();
}
