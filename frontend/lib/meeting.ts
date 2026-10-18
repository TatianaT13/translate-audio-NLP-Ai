import { authFetch } from "@/lib/auth";

const PIPELINE_URL = process.env.NEXT_PUBLIC_PIPELINE_URL || "http://localhost:8000";
const LLM_URL      = process.env.NEXT_PUBLIC_LLM_URL      || "http://localhost:8002";
const STT_URL      = process.env.NEXT_PUBLIC_STT_URL      || "http://localhost:8001";

export interface ChunkTranscript {
  text:          string;
  language:      string;
  confidence:    number;
  duration_s:    number;
}

/** Segment avec identification du speaker (resultat diarisation AssemblyAI). */
export interface DiarizedSegment {
  speaker: string;      // "A", "B", "C"... (ou nom custom apres mapping)
  start:   number;      // secondes
  end:     number;
  text:    string;
}

export interface DiarizedTranscript {
  text:         string;
  language:     string;
  duration:     number;
  segments:     DiarizedSegment[];
  speakers:     string[];
}

/** Envoie un chunk audio (Blob WebM/MP4) au STT pour transcription. */
export async function transcribeChunk(
  audio: Blob,
  whisperModel: string = "small",
  language: string = "auto"       // "auto" = Whisper détecte, "fr" = forcé français…
): Promise<ChunkTranscript> {
  const form = new FormData();
  form.append("file", audio, "chunk.webm");
  form.append("model", whisperModel);
  form.append("language", language);

  const res = await authFetch(`${STT_URL}/transcribe`, {
    method: "POST",
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

/**
 * Transcrit + diarise un audio complet (fusion de tous les chunks d'une reunion).
 * Utilise AssemblyAI cote serveur grace a diarize=true.
 */
export async function transcribeDiarized(
  audio: Blob,
  language: string = "fr",
): Promise<DiarizedTranscript> {
  const form = new FormData();
  form.append("file", audio, "meeting.webm");
  form.append("language", language);
  form.append("diarize", "true");

  const res = await authFetch(`${STT_URL}/transcribe`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`STT diarize erreur ${res.status}`);
  const data = await res.json();
  return {
    text:     data.text || "",
    language: data.language || "",
    duration: data.duration || 0,
    segments: (data.segments || []).map((s: Record<string, unknown>) => ({
      speaker: (s.speaker as string) || "?",
      start:   (s.start as number) || 0,
      end:     (s.end as number) || 0,
      text:    (s.text as string) || "",
    })),
    speakers: data.speakers || [],
  };
}

/**
 * Formate un transcript diarise en texte brut avec balises speaker, pret
 * a etre envoye au LLM pour resume avec attribution par intervenant.
 *   "[Alice] Il faut qu'on revoie...
 *    [Bob]   OK."
 */
export function formatDiarizedTranscript(
  segments: DiarizedSegment[],
  nameMap: Record<string, string> = {},
): string {
  return segments
    .map(s => {
      const name = nameMap[s.speaker] || `Intervenant ${s.speaker}`;
      return `[${name}] ${s.text}`;
    })
    .join("\n");
}

/** Génère un compte-rendu à partir d'un transcript complet. */
export async function summarizeMeeting(
  transcript: string,
  targetLang: string = "en",
  style: "executive" | "detailed" | "actions_only" = "executive"
): Promise<SummaryResponse> {
  const res = await authFetch(`${LLM_URL}/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transcript, target_lang: targetLang, style }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `LLM erreur ${res.status}`);
  }
  return res.json();
}
