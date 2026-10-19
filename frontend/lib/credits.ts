import { authFetch } from "@/lib/auth";

const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8004";

export interface CreditsState {
  processing: {
    balance:       number;
    monthly:       number;
    carryover_max: number;
  };
  live: {
    balance:       number;
    monthly:       number;
    carryover_max: number;
  };
  tier:                  "free" | "pro" | "business";
  trial_credits_granted: boolean;
  period_resets_at:      string | null;
}

export async function getCredits(): Promise<CreditsState | null> {
  const res = await authFetch(`${GATEWAY_URL}/billing/credits`);
  if (!res.ok) return null;
  return res.json();
}

export interface CreditTransaction {
  date:          string | null;
  op:            string;
  amount:        number;
  balance_after: number;
  reason:        string | null;
}

export async function getTransactions(): Promise<CreditTransaction[]> {
  const res = await authFetch(`${GATEWAY_URL}/billing/transactions`);
  if (!res.ok) return [];
  return res.json();
}

/** Libelle lisible d'un mouvement (les raisons sont des codes techniques). */
export function describeTransaction(t: CreditTransaction): string {
  const r = t.reason ?? "";
  if (r === "signup_trial") return "Crédits offerts à l'inscription";
  if (t.op === "monthly_refill" || r.startsWith("tier=")) return "Crédits mensuels de l'abonnement";
  if (t.op === "refund") return "Remboursement";
  if (r.startsWith("meeting_diarization")) return "Identification des intervenants (réunion)";
  if (r.startsWith("meeting_transcription")) return "Transcription de réunion";
  const m = r.match(/^pipeline_process(?:_async)?:(\d+)s:(\w+)/);
  if (m) {
    const min = Math.max(1, Math.round(Number(m[1]) / 60));
    const label = m[2].includes("diarized") ? "Traduction audio avec intervenants" : "Traduction audio";
    return `${label} · ${min} min`;
  }
  return t.amount >= 0 ? "Crédit" : "Consommation";
}

export interface DailyUsage {
  date:          string;   // YYYY-MM-DD
  service:       "translation" | "translation_diarized" | "meeting_transcription" | "meeting_diarization" | "other";
  seconds:       number;
  seconds_known: boolean;  // false : anciens mouvements enregistres sans duree
  credits:       number;
}

export async function getDailyUsage(): Promise<DailyUsage[]> {
  const res = await authFetch(`${GATEWAY_URL}/billing/usage`);
  if (!res.ok) return [];
  return res.json();
}

export const SERVICE_LABEL: Record<DailyUsage["service"], string> = {
  translation:           "Traduction audio",
  translation_diarized:  "Traduction audio avec intervenants",
  meeting_transcription: "Transcription de réunion",
  meeting_diarization:   "Identification des intervenants",
  other:                 "Autre",
};

export function formatDuration(u: DailyUsage): string {
  if (u.seconds <= 0) return "—";
  const total = Math.round(u.seconds);
  const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
  const txt = h ? `${h} h ${String(m).padStart(2, "0")}` : m ? (s ? `${m} min ${s} s` : `${m} min`) : `${s} s`;
  return u.seconds_known ? txt : `≥ ${txt}`;
}
