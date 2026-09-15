"use client";

import { AppHeader } from "@/components/AppHeader";
import { FooterFull } from "@/components/Footer";

interface LegalLayoutProps {
  title:       string;
  updated:     string; // "13 septembre 2026"
  children:    React.ReactNode;
}

export function LegalLayout({ title, updated, children }: LegalLayoutProps) {
  return (
    <>
      <AppHeader variant="landing" />
      <main style={{
        maxWidth: "760px", margin: "0 auto",
        padding: "48px 24px 64px",
      }}>
        <header style={{ marginBottom: "40px" }}>
          <p style={{
            fontSize: "10px", letterSpacing: "0.25em",
            textTransform: "uppercase", color: "var(--muted)",
            margin: "0 0 12px 0", fontWeight: 500,
          }}>
            Informations légales
          </p>
          <h1 style={{
            fontFamily: "var(--font-display), serif",
            fontSize: "clamp(2rem, 4vw, 2.75rem)",
            fontWeight: 400, letterSpacing: "-0.015em",
            color: "var(--foreground)", margin: "0 0 12px 0",
            lineHeight: 1.15,
          }}>
            {title}
          </h1>
          <p style={{
            fontSize: "12px", color: "var(--muted)",
            fontFamily: "ui-monospace, monospace",
            margin: 0,
          }}>
            Dernière mise à jour : {updated}
          </p>
        </header>

        <article style={{ color: "var(--foreground)", fontSize: "15px", lineHeight: 1.7 }}>
          {children}
        </article>
      </main>
      <FooterFull />

      <style>{`
        article h2 {
          font-family: var(--font-display), serif;
          font-size: 1.35rem;
          font-weight: 400;
          color: var(--foreground);
          letter-spacing: -0.005em;
          margin: 40px 0 16px 0;
        }
        article h3 {
          font-size: 1rem;
          font-weight: 600;
          color: var(--foreground);
          margin: 24px 0 8px 0;
        }
        article p { margin: 0 0 14px 0; color: var(--muted); }
        article strong { color: var(--foreground); font-weight: 500; }
        article ul { margin: 0 0 14px 0; padding-left: 24px; color: var(--muted); }
        article li { margin-bottom: 6px; }
        article a { color: var(--accent); text-decoration: none; }
        article a:hover { text-decoration: underline; }
        article .placeholder {
          background: rgba(232, 176, 74, 0.08);
          border: 1px dashed rgba(232, 176, 74, 0.35);
          border-radius: 6px;
          padding: 2px 8px;
          font-family: ui-monospace, monospace;
          font-size: 0.85em;
          color: #e8b04a;
        }
        article dl { margin: 0 0 14px 0; }
        article dt {
          font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase;
          color: var(--muted); font-weight: 500; margin-top: 12px;
        }
        article dd { margin: 4px 0 0 0; color: var(--foreground); }
      `}</style>
    </>
  );
}
