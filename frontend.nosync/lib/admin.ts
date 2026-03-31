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
  bleu_scores:       number[];
  language_probs:    number[];
  latencies_total:   number[];
  model_stats:       LangfuseModelStat[];
}

async function authHeaders(): Promise<HeadersInit> {
  let token = getAccessToken();
  if (!token) token = await refreshAccessToken();
  return token ? { "Authorization": `Bearer ${token}` } : {};
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = await authHeaders();
  const res = await fetch(`${GATEWAY_URL}${path}`, {
    ...init,
    headers: { ...headers, "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
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
