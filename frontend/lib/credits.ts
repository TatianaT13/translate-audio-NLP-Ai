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
