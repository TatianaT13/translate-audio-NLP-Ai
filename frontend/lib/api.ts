import { getAccessToken, refreshAccessToken } from "@/lib/auth";

const PIPELINE_URL = process.env.NEXT_PUBLIC_PIPELINE_URL || "http://localhost:8000";

export interface ProcessResult {
  source_text:        string;
  language:           string;
  language_prob:      number;
  translation:        string;
  audio_b64:          string;
  audio_content_type: string;
  latency_stt_ms:     number;
  latency_llm_ms:     number;
  latency_tts_ms:     number;
  latency_total_ms:   number;
  prompt_tokens?:     number;
  completion_tokens?: number;
  total_tokens?:      number;
  cost_usd?:          number;
  n_chunks?:          number;
  n_tts_chunks?:      number;
}

/** Erreur typée pour un mapping message/action côté UI. */
export class PipelineError extends Error {
  status: number;
  detail: string;
  raw: string;
  constructor(status: number, detail: string, raw = "") {
    super(detail || `Pipeline error ${status}`);
    this.name   = "PipelineError";
    this.status = status;
    this.detail = detail;
    this.raw    = raw;
  }
  /** Message user-friendly selon le code HTTP. */
  get userMessage(): string {
    switch (this.status) {
      case 415: return "Format audio non reconnu. Réencode-le en MP3/WAV via cloudconvert.com puis réessaie.";
      case 413: return "Fichier trop volumineux (limite serveur 100 Mo).";
      case 422: return this.detail || "L'audio n'a pas pu être traité (contenu inhabituel).";
      case 429: return "Trop de requêtes envoyées — attends 30 secondes puis réessaie.";
      case 401: return "Session expirée — reconnecte-toi.";
      case 403: return "Accès refusé.";
      case 502:
      case 503:
      case 504: return "Service temporairement indisponible. Nouvelle tentative en cours…";
      default:  return this.detail || `Erreur serveur (${this.status})`;
    }
  }
  /** Suggestion d'action pour le user. */
  get suggestion(): string | null {
    switch (this.status) {
      case 415: return "Ouvre le lien de conversion en ligne";
      case 422: return "Vérifie que l'audio contient de la parole (pas de la musique/silence)";
      case 413: return "Coupe ton audio en segments plus courts";
      case 429: return "Patiente 30 secondes";
      case 401: return "Reconnecte-toi";
      case 502:
      case 503:
      case 504: return null;  // retry auto
      default:  return null;
    }
  }
}

async function getAuthHeaders(): Promise<HeadersInit> {
  let token = getAccessToken();
  if (!token) token = await refreshAccessToken();
  return token ? { "Authorization": `Bearer ${token}` } : {};
}

export interface RunPipelineOptions {
  targetLang?:    string;
  llmModel?:      string;
  promptVersion?: string;
  whisperModel?:  string;
  signal?:        AbortSignal;   // permet un cancel côté UI
  maxRetries?:    number;         // 502/503/504 uniquement (default: 2)
}

async function _sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(t); reject(new Error("aborted")); });
  });
}

/** Exécute le pipeline avec retry auto sur erreurs transitoires. */
export async function runPipeline(
  file: File | Blob,
  targetLangOrOpts: string | RunPipelineOptions = "en",
  llmModel      = "openai/gpt-4o-mini",
  promptVersion = "v1.1",
  whisperModel  = "small",
): Promise<ProcessResult> {
  // Backward-compat : positional args OU objet options
  const opts: RunPipelineOptions = typeof targetLangOrOpts === "string"
    ? { targetLang: targetLangOrOpts, llmModel, promptVersion, whisperModel }
    : targetLangOrOpts;
  const {
    targetLang    = "en",
    llmModel:     m = "openai/gpt-4o-mini",
    promptVersion: p = "v1.1",
    whisperModel: w  = "small",
    signal,
    maxRetries    = 2,
  } = opts;

  const build = () => {
    const form = new FormData();
    // Utilise le nom de fichier original si dispo (pour le sniffing serveur)
    const filename = "name" in file && file.name ? file.name : "audio.mp3";
    form.append("file",           file, filename);
    form.append("target_lang",    targetLang);
    form.append("llm_model",      m);
    form.append("prompt_version", p);
    form.append("whisper_model",  w);
    return form;
  };

  let lastErr: PipelineError | null = null;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (signal?.aborted) throw new Error("Requête annulée");

    const authHeaders = await getAuthHeaders();
    let res: Response;
    try {
      res = await fetch(`${PIPELINE_URL}/process`, {
        method: "POST",
        headers: authHeaders,
        body: build(),
        signal,
      });
    } catch (e) {
      // Network error / AbortError → propager
      if ((e as Error).name === "AbortError") throw new Error("Requête annulée");
      throw new Error(`Erreur réseau : ${(e as Error).message}`);
    }

    if (res.ok) return res.json();

    // Parse erreur
    const raw = await res.text();
    let detail = "";
    try { detail = JSON.parse(raw).detail || ""; } catch { detail = raw; }
    lastErr = new PipelineError(res.status, detail, raw);

    // Retry uniquement sur erreurs transitoires (502/503/504)
    if (![502, 503, 504].includes(res.status) || attempt >= maxRetries) {
      throw lastErr;
    }
    // Backoff exponentiel : 1s, 2s
    await _sleep(1000 * Math.pow(2, attempt), signal);
  }
  throw lastErr || new Error("Pipeline error inconnue");
}

export function audioFromBase64(b64: string, contentType: string): Blob {
  const binary = atob(b64);
  const bytes  = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: contentType });
}
