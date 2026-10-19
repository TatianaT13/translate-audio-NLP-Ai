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
