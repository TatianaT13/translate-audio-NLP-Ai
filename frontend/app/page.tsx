"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { getMe, refreshAccessToken, logout } from "@/lib/auth";
import type { User } from "@/lib/auth";

// ── Design tokens (aligne sur le reste de l'app) ──────────────────────────────
const C = {
  bg:         "var(--background)",
  surface:    "var(--surface)",
  border:     "var(--border)",
  fg:         "var(--foreground)",
  muted:      "var(--muted)",
  accent:     "var(--accent)",
  accentDim:  "var(--accent-dim)",
};

const S = {
  gap8: "8px", gap12: "12px", gap16: "16px", gap24: "24px", gap32: "32px",
};

// ── Card ──────────────────────────────────────────────────────────────────────
interface FeatureCardProps {
  icon:        React.ReactNode;
  eyebrow:     string;
  title:       string;
  description: string;
  bullets:     string[];
  ctaLabel:    string;
  onClick:     () => void;
  disabled?:   boolean;
  badge?:      string;
}

function FeatureCard({ icon, eyebrow, title, description, bullets, ctaLabel, onClick, disabled, badge }: FeatureCardProps) {
  const [hover, setHover] = useState(false);
  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      onClick={disabled ? undefined : onClick}
      onKeyDown={(e) => { if (!disabled && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); onClick(); } }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: "flex", flexDirection: "column", gap: S.gap16,
        padding: "28px 24px",
        borderRadius: "16px",
        background: C.surface,
        border: `1px solid ${hover && !disabled ? C.accentDim : C.border}`,
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.5 : 1,
        transition: "border-color 0.2s, transform 0.2s, box-shadow 0.2s",
        transform: hover && !disabled ? "translateY(-2px)" : "translateY(0)",
        boxShadow: hover && !disabled ? "0 12px 32px rgba(0,0,0,0.25)" : "none",
        position: "relative",
        minHeight: "340px",
      }}
    >
      {badge && (
        <span style={{
          position: "absolute", top: "14px", right: "14px",
          padding: "3px 8px", borderRadius: "999px",
          background: "rgba(201,169,110,0.12)",
          border: `1px solid ${C.accentDim}`,
          color: C.accent, fontSize: "9px", letterSpacing: "0.15em",
          textTransform: "uppercase", fontFamily: "ui-monospace, monospace",
        }}>{badge}</span>
      )}

      <div style={{ fontSize: "36px", lineHeight: 1, marginBottom: "4px" }}>{icon}</div>

      <div>
        <p style={{
          fontSize: "10px", letterSpacing: "0.22em", textTransform: "uppercase",
          color: C.muted, margin: 0, fontWeight: 500,
        }}>{eyebrow}</p>
        <h2 style={{
          fontFamily: "var(--font-display), serif",
          fontSize: "1.75rem", fontWeight: 400, margin: "8px 0 12px 0",
          color: C.fg, letterSpacing: "-0.01em",
        }}>{title}</h2>
        <p style={{ fontSize: "14px", lineHeight: 1.55, color: C.muted, margin: 0 }}>
          {description}
        </p>
      </div>

      <ul style={{ listStyle: "none", padding: 0, margin: "8px 0 0 0", display: "flex", flexDirection: "column", gap: "6px" }}>
        {bullets.map((b, i) => (
          <li key={i} style={{
            display: "grid", gridTemplateColumns: "12px 1fr", gap: "8px",
            fontSize: "12px", color: C.muted, lineHeight: 1.5,
          }}>
            <span style={{ color: C.accent }}>·</span>
            <span>{b}</span>
          </li>
        ))}
      </ul>

      <div style={{ marginTop: "auto", paddingTop: S.gap16 }}>
        <span style={{
          display: "inline-flex", alignItems: "center", gap: "8px",
          fontSize: "12px", letterSpacing: "0.1em", textTransform: "uppercase",
          color: hover && !disabled ? C.accent : C.fg,
          fontWeight: 500, transition: "color 0.2s",
        }}>
          {ctaLabel}
          <span style={{ fontSize: "14px" }}>→</span>
        </span>
      </div>
    </div>
  );
}

// ── Landing ───────────────────────────────────────────────────────────────────
export default function Landing() {
  const router = useRouter();
  const [user, setUser]       = useState<User | null>(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    getMe()
      .then((u) => setUser(u))
      .catch(() => setUser(null))
      .finally(() => setChecking(false));

    const id = setInterval(() => { refreshAccessToken().catch(() => {}); }, 10 * 60 * 1000);
    return () => clearInterval(id);
  }, []);

  const go = (href: string) => () => {
    if (user) router.push(href);
    else      router.push(`/login?next=${encodeURIComponent(href)}`);
  };

  const handleLogout = async () => {
    await logout().catch(() => {});
    setUser(null);
  };

  return (
    <main style={{ minHeight: "100vh", background: C.bg, color: C.fg, display: "flex", flexDirection: "column" }}>
      {/* ── Nav ── */}
      <header style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "20px 32px",
        borderBottom: `1px solid ${C.border}`,
      }}>
        <Link href="/" style={{
          fontFamily: "var(--font-display), serif",
          fontSize: "18px", color: C.fg, textDecoration: "none",
          letterSpacing: "-0.01em",
        }}>traduction-audio<span style={{ color: C.accent }}>.</span></Link>

        <nav style={{ display: "flex", gap: S.gap16, alignItems: "center" }}>
          {checking ? null : user ? (
            <>
              <span style={{ fontSize: "13px", color: C.muted, fontFamily: "ui-monospace, monospace" }}>
                {user.email}
              </span>
              {user.is_admin && (
                <Link href="/admin" style={{
                  fontSize: "12px", color: C.accent, textDecoration: "none",
                  padding: "6px 12px", borderRadius: "8px",
                  border: `1px solid ${C.accentDim}`,
                  transition: "background 0.15s",
                }}>Admin</Link>
              )}
              <button onClick={handleLogout} style={{
                fontSize: "12px", color: C.muted, background: "none",
                border: "none", cursor: "pointer", padding: "6px 8px",
              }}>Se déconnecter</button>
            </>
          ) : (
            <>
              <Link href="/login" style={{ fontSize: "13px", color: C.muted, textDecoration: "none" }}>
                Se connecter
              </Link>
              <Link href="/register" style={{
                fontSize: "13px", color: C.accent, textDecoration: "none",
                padding: "8px 16px", borderRadius: "8px",
                border: `1px solid ${C.accentDim}`,
                background: "rgba(201,169,110,0.06)",
              }}>Créer un compte</Link>
            </>
          )}
        </nav>
      </header>

      {/* ── Hero ── */}
      <section style={{
        padding: "80px 32px 40px", textAlign: "center",
        maxWidth: "720px", margin: "0 auto",
      }}>
        <p style={{
          fontSize: "11px", letterSpacing: "0.25em", textTransform: "uppercase",
          color: C.muted, margin: "0 0 24px 0",
        }}>Plateforme LLMOps · Traduction audio</p>

        <h1 style={{
          fontFamily: "var(--font-display), serif",
          fontSize: "clamp(2.5rem, 5vw, 4rem)", fontWeight: 400,
          lineHeight: 1.1, letterSpacing: "-0.02em", margin: "0 0 24px 0",
        }}>
          Parlez<span style={{ color: C.accent }}>.</span><br />
          <em style={{ color: C.accent }}>On traduit.</em>
        </h1>

        <p style={{
          fontSize: "16px", lineHeight: 1.6, color: C.muted,
          maxWidth: "540px", margin: "0 auto",
        }}>
          Traduction audio propulsée par Whisper, GPT-4o mini et Voxtral.
          Choisissez le mode adapté à votre besoin.
        </p>
      </section>

      {/* ── 3 Cards ── */}
      <section style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
        gap: S.gap24,
        padding: "20px 32px 80px",
        maxWidth: "1200px", margin: "0 auto", width: "100%",
      }}>
        <FeatureCard
          icon="📁"
          eyebrow="Cas d'usage principal"
          title="Traduire un audio"
          description="Uploadez un fichier ou enregistrez-vous. On transcrit, on traduit, on synthétise à voix haute."
          bullets={[
            "MP3, WAV, M4A, OGG, WebM, FLAC · max 25 Mo",
            "Anglais, ukrainien, espagnol, allemand",
            "Transcription + traduction + audio synthétisé",
          ]}
          ctaLabel={user ? "Ouvrir" : "Se connecter"}
          onClick={go("/translate")}
        />

        <FeatureCard
          icon="🎙"
          eyebrow="Temps réel · bêta"
          title="Live speech-to-speech"
          description="Parlez en français, entendez la réponse dans la langue cible instantanément. Latence sub-seconde."
          bullets={[
            "WebRTC · OpenAI Realtime API",
            "6 langues source × 6 cibles",
            "Push-to-talk ou VAD automatique",
          ]}
          ctaLabel={user ? "Ouvrir" : "Se connecter"}
          onClick={go("/live")}
          badge="Bêta"
        />

        <FeatureCard
          icon="📝"
          eyebrow="Réunions multilingues"
          title="Compte-rendu de réunion"
          description="Enregistrez une réunion longue. On génère un résumé structuré : synthèse, actions, décisions."
          bullets={[
            "Enregistrement continu (>30 min)",
            "Résumé exécutif · détaillé · actions",
            "Export texte, dans 5 langues",
          ]}
          ctaLabel={user ? "Ouvrir" : "Se connecter"}
          onClick={go("/meeting")}
        />
      </section>

      {/* ── Footer ── */}
      <footer style={{
        marginTop: "auto", padding: "24px 32px",
        borderTop: `1px solid ${C.border}`, textAlign: "center",
      }}>
        <p style={{ fontSize: "11px", letterSpacing: "0.12em", color: C.muted, opacity: 0.5, margin: 0 }}>
          © {new Date().getFullYear()} traduction-audio.fr · Whisper · Llama · Voxtral
        </p>
      </footer>
    </main>
  );
}
