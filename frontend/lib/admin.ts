import { getAccessToken, refreshAccessToken } from "@/lib/auth";

const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8004";

export interface AdminUser {
  id:         number;
  email:      string;
  is_active:  boolean;
  is_admin:   boolean;
  created_at: string | null;
}

export interface AdminStats {
  total_users:  number;
  active_users: number;
  admin_users:  number;
}

export interface LangfuseModelStat {
  whisper:        string;
  llm:            string;
  prompt_version: string;
  count:          number;
  avg_total_ms:   number;
  avg_stt_ms:     number;
  avg_llm_ms:     number;
  avg_bleu:       number | null;
  avg_cost_usd?:  number | null;
  avg_tokens?:    number | null;
  avg_meteor:     number | null;
  avg_wer:        number | null;
}

export interface LangfuseMetrics {
  connected:         boolean;
  error?:            string;
  total_traces:      number;
  avg_total_ms:      number;
  avg_stt_ms:        number;
  avg_llm_ms:        number;
  avg_language_prob: number;
  avg_bleu:          number;
  avg_meteor:        number;
  avg_wer:           number;
  avg_cost_usd?:     number;
  total_cost_usd?:   number;
  avg_tokens?:       number;
  total_tokens?:     number;
  bleu_scores:       number[];
  meteor_scores:     number[];
  wer_scores:        number[];
  language_probs:    number[];
  latencies_total:   number[];
  cost_scores?:      number[];
  model_stats:       LangfuseModelStat[];
}

async function authHeaders(): Promise<HeadersInit> {
  let token = getAccessToken();
  if (!token) token = await refreshAccessToken();
  return token ? { "Authorization": `Bearer ${token}` } : {};
}

async function apiFetch<T>(path: string, init?: RequestInit, _retried = false): Promise<T> {
  const headers = await authHeaders();
  const res = await fetch(`${GATEWAY_URL}${path}`, {
    ...init,
    headers: { ...headers, "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });

  // Token expiré → refresh + retry une fois
  if (res.status === 401 && !_retried) {
    const newToken = await refreshAccessToken();
    if (newToken) return apiFetch<T>(path, init, true);
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Erreur ${res.status}`);
  }
  return res.json();
}

export const getAdminStats    = () => apiFetch<AdminStats>("/admin/stats");
export const getAdminUsers    = () => apiFetch<AdminUser[]>("/admin/users");
export const getLangfuseMetrics = () => apiFetch<LangfuseMetrics>("/admin/langfuse/metrics");

export const updateAdminUser = (id: number, body: { is_active?: boolean; is_admin?: boolean }) =>
  apiFetch(`/admin/users/${id}`, { method: "PATCH", body: JSON.stringify(body) });

export const deleteAdminUser = (id: number) =>
  apiFetch(`/admin/users/${id}`, { method: "DELETE" });

export const seedAdmin = () =>
  apiFetch("/admin/seed", { method: "POST" });

export interface MlflowExperiment {
  id:   string;
  name: string;
  runs: number;
}
export interface MlflowModel {
  name:               string;
  description:        string;
  production_version: string;
  provider:           string;
  type:               string;
}
export interface MlflowSummary {
  connected:   boolean;
  url:         string;
  experiments: MlflowExperiment[];
  models:      MlflowModel[];
  total_runs:  number;
  error?:      string | null;
}
export const getMlflowSummary = () =>
  apiFetch<MlflowSummary>("/admin/mlflow/summary");

export interface AirflowDag {
  dag_id:      string;
  description: string;
  schedule:    string | null;
  is_paused:   boolean;
  tags:        string[];
  last_run:    { state: string; start_date: string; end_date: string | null } | null;
}
export interface AirflowSummary {
  connected: boolean;
  url:       string;
  dags:      AirflowDag[];
  error?:    string | null;
}
export const getAirflowSummary = () =>
  apiFetch<AirflowSummary>("/admin/airflow/summary");

export interface ServiceHealth {
  name:       string;
  port:       string;
  color:      string;
  status:     "up" | "down" | "error";
  latency_ms: number;
  detail:     Record<string, unknown>;
}

export const getServicesHealth = () =>
  apiFetch<{ services: ServiceHealth[] }>("/admin/services/health");

export interface TrafficEvent {
  type:          string;          // type principal (le plus sévère)
  types?:        string[];        // tous les types détectés sur cette portion (fusion)
  severity:      "high" | "medium" | "low";
  routes:        string[];
  direction:     string;
  location_hint: string;
  zone:          string;
  timestamp:     string;
  delay_hint:    string;
  translations?: Record<string, string>;
}

export async function synthesizeTTS(text: string, lang: string): Promise<Blob> {
  const headers = await authHeaders();
  const res = await fetch(`${GATEWAY_URL}/admin/tts`, {
    method: "POST",
    headers: { ...headers, "Content-Type": "application/json" },
    body: JSON.stringify({ text, lang }),
  });
  if (!res.ok) throw new Error(`TTS erreur ${res.status}`);
  return res.blob();
}

export type TrafficSnapshot = Record<"nord" | "sud" | "ouest", TrafficEvent[]>;

export const getTrafficEvents = () =>
  apiFetch<TrafficSnapshot>("/admin/traffic/events");

export interface ExperimentRun {
  run_id:           string;
  audio:            string;
  zone:             string;
  whisper_model:    string;
  llm_model:        string;
  prompt_version:   string;
  target_lang:      string;
  language_prob:    number | null;
  latency_stt_ms:   number | null;
  latency_llm_ms:   number | null;
  latency_total_ms: number | null;
  bleu:             number | null;
  meteor:           number | null;
  wer:              number | null;
  tts_wer:          number | null;
}

export interface ExperimentsResponse {
  runs:       ExperimentRun[];
  total:      number;
  csv_exists: boolean;
}

export const getExperiments = () =>
  apiFetch<ExperimentsResponse>("/admin/experiments");

export function openTrafficStream(onEvent: (data: TrafficSnapshot | { zone: string; events: TrafficEvent[] }) => void): EventSource {
  const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8004";
  const token = getAccessToken() ?? "";
  const es = new EventSource(`${GATEWAY}/admin/traffic/stream?token=${encodeURIComponent(token)}`);
  es.onmessage = (e) => {
    try { onEvent(JSON.parse(e.data)); } catch {}
  };
  return es;
}
