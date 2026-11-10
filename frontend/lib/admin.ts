import { authFetch } from "@/lib/auth";

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

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  // authFetch : cookies de session, renouvellement automatique sur 401
  const res = await authFetch(`${GATEWAY_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Erreur ${res.status}`);
  }
  return res.json();
}

export const getAdminStats    = () => apiFetch<AdminStats>("/admin/stats");
export const getAdminUsers    = () => apiFetch<AdminUser[]>("/admin/users");

// Marge par service : crédits débités (revenu) contre coût fournisseur réel.
export interface MarginRow {
  service:           string;
  seconds:           number;
  credits:           number;
  untracked_credits: number;   // débits antérieurs au suivi : coût inconnu
  revenue_eur:       number;
  cost_eur:          number;
  margin_pct:        number | null;
}

export interface MarginsReport {
  days:             number;
  credit_value_eur: number;
  usd_to_eur:       number;
  services:         MarginRow[];
  total:            MarginRow;
}

export const getAdminMargins = (days = 30) => apiFetch<MarginsReport>(`/admin/margins?days=${days}`);
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

// TrafficEvent + TrafficSnapshot + getTrafficEvents retires avec la
// desactivation du service watcher (v0.2 — mise en production commerciale).

export async function synthesizeTTS(text: string, lang: string): Promise<Blob> {
  const res = await authFetch(`${GATEWAY_URL}/admin/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, lang }),
  });
  if (!res.ok) throw new Error(`TTS erreur ${res.status}`);
  return res.blob();
}

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

// openTrafficStream retire avec la desactivation du service watcher
