import { authFetch } from "@/lib/auth";
import type { TranslationSegment } from "@/lib/api";

const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8004";

export type HistoryKind = "translation" | "meeting";

export interface HistoryMeta {
  id:         string;
  kind:       HistoryKind;
  title:      string;
  lang:       string | null;
  duration_s: number | null;
  created_at: string | null;
  expires_at: string | null;
}

export interface TranslationContent {
  source_text: string;
  translation: string;
  language?:   string;
  segments?:   TranslationSegment[] | null;   // avec intervenants, si demandé
}

export interface MeetingContent {
  transcript:     string;
  summary:        string;
  summary_style?: string;
}

export interface HistoryEntry {
  kind:        HistoryKind;
  title:       string;
  lang?:       string | null;
  duration_s?: number | null;
  content:     TranslationContent | MeetingContent;
}

export type HistoryItem = HistoryMeta & { content: TranslationContent | MeetingContent };

/**
 * Enregistre un résultat dans l'historique. Ne lève jamais : l'historique est
 * un confort, son échec ne doit pas gâcher un traitement réussi.
 */
export async function saveHistory(entry: HistoryEntry): Promise<HistoryMeta | null> {
  try {
    const res = await authFetch(`${GATEWAY_URL}/history`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    });
    return res.ok ? res.json() : null;
  } catch {
    return null;
  }
}

/** Remplace un élément existant (compte-rendu régénéré). Ne lève jamais. */
export async function updateHistory(id: string, entry: HistoryEntry): Promise<void> {
  try {
    await authFetch(`${GATEWAY_URL}/history/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    });
  } catch {}
}

export async function listHistory(): Promise<{ retention_days: number; items: HistoryMeta[] }> {
  const res = await authFetch(`${GATEWAY_URL}/history`);
  if (!res.ok) throw new Error("Historique indisponible");
  return res.json();
}

export async function getHistoryItem(id: string): Promise<HistoryItem> {
  const res = await authFetch(`${GATEWAY_URL}/history/${id}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Élément introuvable");
  }
  return res.json();
}

export async function deleteHistoryItem(id: string): Promise<void> {
  const res = await authFetch(`${GATEWAY_URL}/history/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Suppression impossible");
}

export async function clearHistory(): Promise<void> {
  const res = await authFetch(`${GATEWAY_URL}/history`, { method: "DELETE" });
  if (!res.ok) throw new Error("Suppression impossible");
}

export function isMeeting(content: TranslationContent | MeetingContent): content is MeetingContent {
  return "summary" in content;
}

/** Version texte d'un élément, pour le copier ou le télécharger. */
export function historyAsText(item: HistoryItem): string {
  const c = item.content;
  if (isMeeting(c)) {
    return `${item.title}\n\nCOMPTE-RENDU\n\n${c.summary}\n\nTRANSCRIPTION\n\n${c.transcript}\n`;
  }
  if (c.segments?.length) {
    const lines = c.segments.map((s) => `[${s.speaker}] ${s.original}\n→ ${s.translation}`);
    return `${item.title}\n\n${lines.join("\n\n")}\n`;
  }
  return `${item.title}\n\nTEXTE D'ORIGINE\n\n${c.source_text}\n\nTRADUCTION\n\n${c.translation}\n`;
}
