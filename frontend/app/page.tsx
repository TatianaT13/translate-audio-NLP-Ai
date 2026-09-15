"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getMe } from "@/lib/auth";
import type { User } from "@/lib/auth";
import { AppHeader } from "@/components/AppHeader";
import { TranslateIcon, MicIcon, DocumentIcon } from "@/components/icons";

// ── Card ──────────────────────────────────────────────────────────────────────
interface FeatureCardProps {
  Icon:        React.FC<{ size?: number }>;
  eyebrow:     string;
  title:       string;
  description: string;
  bullets:     string[];
  ctaLabel:    string;
  onClick:     () => void;
  badge?:      string;
}

function FeatureCard({ Icon, eyebrow, title, description, bullets, ctaLabel, onClick, badge }: FeatureCardProps) {
  const [hover, setHover] = useState(false);
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); } }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: "flex", flexDirection: "column", gap: "16px",
        padding: "28px 24px",
        borderRadius: "16px",
        background: "var(--surface)",
        border: `1px solid ${hover ? "var(--accent-dim)" : "var(--border)"}`,
        cursor: "pointer",
        transition: "border-color 0.2s, transform 0.2s, box-shadow 0.2s",
        transform: hover ? "translateY(-2px)" : "translateY(0)",
        boxShadow: hover ? "0 12px 32px rgba(0,0,0,0.25)" : "none",
        position: "relative",
        minHeight: "340px",
      }}
    >
      {badge && (
        <span style={{
          position: "absolute", top: "14px", right: "14px",
          padding: "3px 8px", borderRadius: "999px",
          background: "rgba(201,169,110,0.12)",
          border: "1px solid var(--accent-dim)",
          color: "var(--accent)", fontSize: "9px", letterSpacing: "0.15em",
          textTransform: "uppercase", fontFamily: "ui-monospace, monospace",
        }}>{badge}</span>
      )}

      {/* Icone */}
      <div style={{
        width: "56px", height: "56px",
        borderRadius: "14px",
        background: "rgba(201,169,110,0.08)",
        border: "1px solid var(--accent-dim)",
        display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--accent)",
        transition: "background 0.2s, border-color 0.2s",
      }}>
        <Icon size={28} />
      </div>

      <div>
        <p style={{
          fontSize: "10px", letterSpacing: "0.22em", textTransform: "uppercase",
          color: "var(--muted)", margin: 0, fontWeight: 500,
        }}>{eyebrow}</p>
        <h2 style={{
          fontFamily: "var(--font-display), serif",
          fontSize: "1.75rem", fontWeight: 400, margin: "8px 0 12px 0",
          color: "var(--foreground)", letterSpacing: "-0.01em",
        }}>{title}</h2>
        <p style={{ fontSize: "14px", lineHeight: 1.55, color: "var(--muted)", margin: 0 }}>
          {description}
        </p>
      </div>

      <ul style={{ listStyle: "none", padding: 0, margin: "8px 0 0 0", display: "flex", flexDirection: "column", gap: "6px" }}>
        {bullets.map((b, i) => (
          <li key={i} style={{
            display: "grid", gridTemplateColumns: "12px 1fr", gap: "8px",
            fontSize: "12px", color: "var(--muted)", lineHeight: 1.5,
          }}>
            <span style={{ color: "var(--accent)" }}>·</span>
            <span>{b}</span>
          </li>
        ))}
      </ul>

      <div style={{ marginTop: "auto", paddingTop: "16px" }}>
        <span style={{
          display: "inline-flex", alignItems: "center", gap: "8px",
          fontSize: "12px", letterSpacing: "0.1em", textTransform: "uppercase",
          color: hover ? "var(--accent)" : "var(--foreground)",
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
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    getMe().then(setUser).catch(() => setUser(null));
  }, []);

  const go = (href: string) => () => {
    if (user) router.push(href);
    else      router.push(`/login?next=${encodeURIComponent(href)}`);
  };

  return (
    <main style={{ minHeight: "100vh", background: "var(--background)", color: "var(--foreground)", display: "flex", flexDirection: "column" }}>
      <AppHeader variant="landing" />

      {/* ── Hero ── */}
      <section style={{
        padding: "80px 32px 40px", textAlign: "center",
        maxWidth: "720px", margin: "0 auto",
      }}>
        <p style={{
          fontSize: "11px", letterSpacing: "0.25em", textTransform: "uppercase",
          color: "var(--muted)", margin: "0 0 24px 0",
        }}>Plateforme LLMOps · Traduction audio</p>

        <h1 style={{
          fontFamily: "var(--font-display), serif",
          fontSize: "clamp(2.5rem, 5vw, 4rem)", fontWeight: 400,
          lineHeight: 1.1, letterSpacing: "-0.02em", margin: "0 0 24px 0",
        }}>
          Parlez<span style={{ color: "var(--accent)" }}>.</span><br />
          <em style={{ color: "var(--accent)" }}>On traduit.</em>
        </h1>

        <p style={{
          fontSize: "16px", lineHeight: 1.6, color: "var(--muted)",
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
        gap: "24px",
        padding: "20px 32px 80px",
        maxWidth: "1200px", margin: "0 auto", width: "100%",
      }}>
        <FeatureCard
          Icon={TranslateIcon}
          eyebrow="Cas d'usage principal"
          title="Traduire un audio"
          description="Uploadez un fichier ou enregistrez-vous. On transcrit, on traduit, on synthétise à voix haute."
          bullets={[
            "MP3, WAV, M4A, OGG, WebM, FLAC · max 25 Mo",
            "Anglais, ukrainien, espagnol, allemand",
            "Whisper large-v3 + GPT-4o mini + Voxtral",
          ]}
          ctaLabel={user ? "Ouvrir" : "Se connecter"}
          onClick={go("/translate")}
        />

        <FeatureCard
          Icon={MicIcon}
          eyebrow="Temps réel · bêta"
          title="Live speech-to-speech"
          description="Parlez en français, entendez la réponse dans la langue cible instantanément. Latence sub-seconde."
          bullets={[
            "WebRTC · OpenAI Realtime API",
            "6 langues source × 6 cibles",
            "VAD automatique ou push-to-talk",
          ]}
          ctaLabel={user ? "Ouvrir" : "Se connecter"}
          onClick={go("/live")}
          badge="Bêta"
        />

        <FeatureCard
          Icon={DocumentIcon}
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
        borderTop: "1px solid var(--border)", textAlign: "center",
      }}>
        <p style={{ fontSize: "11px", letterSpacing: "0.12em", color: "var(--muted)", opacity: 0.5, margin: 0 }}>
          © {new Date().getFullYear()} traduction-audio.fr · Whisper · Llama · Voxtral
        </p>
      </footer>
    </main>
  );
}
