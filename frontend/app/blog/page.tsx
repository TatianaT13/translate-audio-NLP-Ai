import type { Metadata } from "next";
import Link from "next/link";
import { AppHeader } from "@/components/AppHeader";
import { FooterFull } from "@/components/Footer";
import { ARTICLES } from "@/lib/blog";

export const metadata: Metadata = {
  title: "Blog — Guides sur la transcription et la traduction audio par IA",
  description: "Guides pratiques, comparatifs et tutoriels pour transcrire, traduire et sous-titrer vos audios et vidéos avec l'IA.",
  alternates: { canonical: "/blog" },
  openGraph: {
    title: "Blog traduction-audio.fr — Guides IA pour l'audio",
    description: "Transcription, traduction, diarisation, sous-titrage : nos guides et comparatifs.",
    url: "/blog",
  },
};

const CATEGORY_LABEL: Record<string, string> = {
  guide:      "Guide",
  comparatif: "Comparatif",
  tutoriel:   "Tutoriel",
};

export default function BlogIndex() {
  return (
    <>
      <AppHeader variant="landing" />
      <main style={{ maxWidth: "860px", margin: "0 auto", padding: "48px 24px 64px" }}>
        <header style={{ marginBottom: "48px", textAlign: "center" }}>
          <p style={{
            fontSize: "10px", letterSpacing: "0.25em", textTransform: "uppercase",
            color: "var(--accent)", margin: "0 0 12px", fontWeight: 500,
          }}>
            Blog traduction-audio.fr
          </p>
          <h1 style={{
            fontFamily: "var(--font-display), serif",
            fontSize: "clamp(2rem, 5vw, 3rem)", fontWeight: 400,
            color: "var(--foreground)", margin: "0 0 16px", lineHeight: 1.15,
            letterSpacing: "-0.015em",
          }}>
            Guides IA pour transcrire, traduire, sous-titrer
          </h1>
          <p style={{ fontSize: "15px", color: "var(--muted)", maxWidth: "600px", margin: "0 auto", lineHeight: 1.6 }}>
            Méthodes et comparatifs d&apos;outils pour tirer le meilleur de
            l&apos;intelligence artificielle sur vos audios et vidéos.
          </p>
        </header>

        <div style={{ display: "grid", gap: "24px" }}>
          {ARTICLES.map(a => {
            const dateFr = new Date(a.date).toLocaleDateString("fr-FR", {
              year: "numeric", month: "long", day: "numeric",
            });
            return (
              <Link
                key={a.slug}
                href={`/blog/${a.slug}`}
                style={{
                  display: "block", textDecoration: "none",
                  padding: "24px 28px", borderRadius: "16px",
                  background: "var(--surface)", border: "1px solid var(--border)",
                  transition: "border-color 0.2s, transform 0.2s",
                }}
              >
                <div style={{
                  display: "flex", gap: "12px", fontSize: "10px",
                  letterSpacing: "0.2em", textTransform: "uppercase",
                  color: "var(--accent)", marginBottom: "10px", fontWeight: 600,
                }}>
                  <span>{CATEGORY_LABEL[a.category] || a.category}</span>
                  <span style={{ color: "var(--muted)" }}>·</span>
                  <span style={{ color: "var(--muted)" }}>{a.readMinutes} min</span>
                </div>
                <h2 style={{
                  fontFamily: "var(--font-display), serif", fontSize: "1.5rem",
                  fontWeight: 400, color: "var(--foreground)", margin: "0 0 10px",
                  lineHeight: 1.3,
                }}>
                  {a.title}
                </h2>
                <p style={{ margin: "0 0 12px", color: "var(--muted)", fontSize: "14px", lineHeight: 1.6 }}>
                  {a.excerpt}
                </p>
                <p style={{ margin: 0, fontSize: "11px", color: "var(--muted)", fontFamily: "ui-monospace, monospace", opacity: 0.7 }}>
                  {dateFr}
                </p>
              </Link>
            );
          })}
        </div>
      </main>
      <FooterFull />
    </>
  );
}
