"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getMe, authFetch } from "@/lib/auth";
import type { User } from "@/lib/auth";
import { AppHeader } from "@/components/AppHeader";
import { FooterFull } from "@/components/Footer";

const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8004";

interface Plan {
  key:        "free" | "monthly" | "yearly";
  name:       string;
  price:      string;
  pricePer:   string;
  savings?:   string;
  bullets:    string[];
  cta:        string;
  highlight?: boolean;
}

const PLANS: Plan[] = [
  {
    key: "free",
    name: "Découverte",
    price: "0 €",
    pricePer: "à vie",
    bullets: [
      "10 traductions audio par mois",
      "Fichiers jusqu'à 25 Mo",
      "5 langues au choix",
      "Compte-rendu de réunion (essai)",
    ],
    cta: "Commencer gratuitement",
  },
  {
    key: "monthly",
    name: "Pro",
    price: "9,90 €",
    pricePer: "par mois",
    bullets: [
      "200 traductions par mois",
      "Fichiers jusqu'à 100 Mo",
      "Toutes les langues",
      "Traduction live en temps réel",
      "Compte-rendus de réunion illimités",
      "Support prioritaire",
    ],
    cta: "S'abonner mensuel",
    highlight: true,
  },
  {
    key: "yearly",
    name: "Pro annuel",
    price: "99 €",
    pricePer: "par an",
    savings: "2 mois offerts",
    bullets: [
      "Tout ce qui est dans Pro mensuel",
      "Paiement annuel unique",
      "2 mois offerts vs mensuel",
      "Facturation annuelle simplifiée",
    ],
    cta: "S'abonner annuel",
  },
];

export default function TarifsPage() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMe().then(setUser).catch(() => setUser(null));
  }, []);

  const subscribe = async (plan: "monthly" | "yearly") => {
    if (!user) {
      router.push(`/login?next=${encodeURIComponent("/tarifs")}`);
      return;
    }
    setError(null);
    setLoading(plan);
    try {
      const res = await authFetch(`${GATEWAY_URL}/billing/checkout`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Erreur ${res.status}`);
      }
      const data = await res.json();
      if (data.url) window.location.href = data.url;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur de paiement");
      setLoading(null);
    }
  };

  const startFree = () => {
    if (user) router.push("/translate");
    else      router.push("/register?next=/translate");
  };

  return (
    <>
      <AppHeader variant="landing" />

      <main style={{
        minHeight: "calc(100vh - 65px)",
        padding: "56px 24px 80px",
        background: "var(--background)",
      }}>
        <div style={{ maxWidth: "1200px", margin: "0 auto" }}>

          {/* Hero */}
          <div style={{ textAlign: "center", marginBottom: "56px" }}>
            <p style={{
              fontSize: "11px", letterSpacing: "0.25em",
              textTransform: "uppercase", color: "var(--muted)",
              margin: "0 0 16px 0", fontWeight: 500,
            }}>
              Tarifs
            </p>
            <h1 style={{
              fontFamily: "var(--font-display), serif",
              fontSize: "clamp(2.25rem, 5vw, 3.5rem)",
              fontWeight: 400, letterSpacing: "-0.02em",
              margin: "0 0 16px 0", lineHeight: 1.1,
              color: "var(--foreground)",
            }}>
              Choisissez votre formule<span style={{ color: "var(--accent)" }}>.</span>
            </h1>
            <p style={{
              fontSize: "15px", lineHeight: 1.6,
              color: "var(--muted)", maxWidth: "540px", margin: "0 auto",
            }}>
              Commencez gratuitement. Passez au plan Pro dès que vous en avez besoin.
              Sans engagement, résiliable à tout moment.
            </p>
          </div>

          {/* Cards */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: "24px",
            maxWidth: "1080px", margin: "0 auto",
          }}>
            {PLANS.map((plan) => (
              <PlanCard
                key={plan.key}
                plan={plan}
                loading={loading === plan.key}
                onSubscribe={
                  plan.key === "free"
                    ? startFree
                    : () => subscribe(plan.key as "monthly" | "yearly")
                }
              />
            ))}
          </div>

          {error && (
            <p style={{
              marginTop: "24px", textAlign: "center",
              padding: "12px 16px", borderRadius: "10px",
              background: "rgba(232,112,112,0.08)",
              border: "1px solid rgba(232,112,112,0.25)",
              color: "#e87070", fontSize: "13px",
              maxWidth: "480px", marginLeft: "auto", marginRight: "auto",
            }}>{error}</p>
          )}

          {/* Notes */}
          <div style={{ textAlign: "center", marginTop: "56px", maxWidth: "640px", marginLeft: "auto", marginRight: "auto" }}>
            <p style={{ fontSize: "12px", color: "var(--muted)", lineHeight: 1.6 }}>
              Paiement sécurisé par <strong>Stripe</strong>. Vos données bancaires ne transitent pas
              par nos serveurs. Résiliation en un clic depuis votre espace personnel.
              TVA non applicable, art. 293 B du CGI.
            </p>
            <p style={{ fontSize: "12px", color: "var(--muted)", marginTop: "12px" }}>
              Une question ? <a href="/contact" style={{ color: "var(--accent)", textDecoration: "none" }}>Contactez-nous</a>
            </p>
          </div>
        </div>
      </main>

      <FooterFull />
    </>
  );
}

// ── Plan Card ────────────────────────────────────────────────────────────────
function PlanCard({ plan, loading, onSubscribe }: {
  plan: Plan;
  loading: boolean;
  onSubscribe: () => void;
}) {
  const [hover, setHover] = useState(false);

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: "flex", flexDirection: "column",
        padding: "32px 24px",
        borderRadius: "16px",
        background: "var(--surface)",
        border: `1px solid ${plan.highlight ? "var(--accent-dim)" : "var(--border)"}`,
        boxShadow: plan.highlight ? "0 12px 40px rgba(201,169,110,0.10)" : (hover ? "0 8px 24px rgba(0,0,0,0.2)" : "none"),
        transform: hover ? "translateY(-2px)" : "translateY(0)",
        transition: "all 0.2s",
        position: "relative",
        minHeight: "480px",
      }}
    >
      {plan.highlight && (
        <span style={{
          position: "absolute", top: "-11px", left: "50%", transform: "translateX(-50%)",
          padding: "4px 12px", borderRadius: "999px",
          background: "linear-gradient(135deg, var(--accent), var(--accent-dim))",
          color: "#0c0c0e", fontSize: "10px", letterSpacing: "0.2em",
          textTransform: "uppercase", fontWeight: 600,
        }}>Recommandé</span>
      )}
      {plan.savings && (
        <span style={{
          position: "absolute", top: "14px", right: "14px",
          padding: "3px 8px", borderRadius: "999px",
          background: "rgba(126,201,160,0.10)",
          border: "1px solid rgba(126,201,160,0.35)",
          color: "#7ec9a0", fontSize: "9px", letterSpacing: "0.15em",
          textTransform: "uppercase",
        }}>{plan.savings}</span>
      )}

      {/* Nom */}
      <h2 style={{
        fontFamily: "var(--font-display), serif",
        fontSize: "1.5rem", fontWeight: 400,
        color: "var(--foreground)", margin: "0 0 8px 0",
        letterSpacing: "-0.005em",
      }}>{plan.name}</h2>

      {/* Prix */}
      <div style={{ display: "flex", alignItems: "baseline", gap: "8px", marginBottom: "24px" }}>
        <span style={{
          fontFamily: "var(--font-display), serif",
          fontSize: "clamp(2rem, 4vw, 2.75rem)", fontWeight: 400,
          color: plan.highlight ? "var(--accent)" : "var(--foreground)",
          letterSpacing: "-0.02em",
        }}>{plan.price}</span>
        <span style={{ fontSize: "13px", color: "var(--muted)" }}>{plan.pricePer}</span>
      </div>

      {/* Bullets */}
      <ul style={{ listStyle: "none", padding: 0, margin: "0 0 24px 0", display: "flex", flexDirection: "column", gap: "10px" }}>
        {plan.bullets.map((b, i) => (
          <li key={i} style={{
            display: "grid", gridTemplateColumns: "16px 1fr", gap: "8px",
            fontSize: "13px", color: "var(--foreground)", lineHeight: 1.5,
          }}>
            <span style={{ color: "var(--accent)", fontSize: "14px", lineHeight: 1.4 }}>✓</span>
            <span>{b}</span>
          </li>
        ))}
      </ul>

      {/* CTA — pousse en bas */}
      <div style={{ marginTop: "auto" }}>
        <button
          onClick={onSubscribe}
          disabled={loading}
          style={{
            width: "100%", padding: "14px", borderRadius: "12px",
            fontSize: "14px", fontWeight: 500, letterSpacing: "0.02em",
            cursor: loading ? "wait" : "pointer",
            background: plan.highlight
              ? "linear-gradient(135deg, var(--accent), var(--accent-dim))"
              : "var(--surface)",
            color: plan.highlight ? "#0c0c0e" : "var(--foreground)",
            border: plan.highlight ? "none" : "1px solid var(--border)",
            transition: "all 0.15s",
            opacity: loading ? 0.7 : 1,
          }}
        >
          {loading ? "Redirection…" : plan.cta}
        </button>
      </div>
    </div>
  );
}
