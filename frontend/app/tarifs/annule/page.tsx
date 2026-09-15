import Link from "next/link";
import { AppHeader } from "@/components/AppHeader";
import { FooterMinimal } from "@/components/Footer";

export const metadata = { title: "Paiement annulé — traduction-audio.fr" };

export default function AnnulePage() {
  return (
    <>
      <AppHeader variant="landing" />
      <main style={{
        minHeight: "calc(100vh - 65px)",
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        padding: "24px", background: "var(--background)",
      }}>
        <div style={{ maxWidth: "480px", textAlign: "center" }}>

          <div style={{
            width: "64px", height: "64px", borderRadius: "50%",
            background: "rgba(138,138,138,0.10)",
            border: "1px solid var(--border)",
            display: "flex", alignItems: "center", justifyContent: "center",
            margin: "0 auto 24px",
          }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
          </div>

          <h1 style={{
            fontFamily: "var(--font-display), serif",
            fontSize: "2rem", fontWeight: 400,
            color: "var(--foreground)", margin: "0 0 12px 0",
            letterSpacing: "-0.015em",
          }}>
            Paiement annulé
          </h1>

          <p style={{
            fontSize: "15px", lineHeight: 1.6,
            color: "var(--muted)", margin: "0 0 32px 0",
          }}>
            Aucun montant n&apos;a été prélevé. Vous pouvez revenir à la page des tarifs
            ou continuer à utiliser la formule Découverte gratuitement.
          </p>

          <div style={{ display: "flex", gap: "12px", justifyContent: "center", flexWrap: "wrap" }}>
            <Link href="/tarifs" style={{
              padding: "12px 24px", borderRadius: "12px",
              background: "linear-gradient(135deg, var(--accent), var(--accent-dim))",
              color: "#0c0c0e", fontSize: "14px", fontWeight: 500,
              textDecoration: "none",
            }}>Revoir les tarifs</Link>
            <Link href="/translate" style={{
              padding: "12px 24px", borderRadius: "12px",
              background: "var(--surface)", border: "1px solid var(--border)",
              color: "var(--muted)", fontSize: "14px",
              textDecoration: "none",
            }}>Continuer en gratuit</Link>
          </div>
        </div>
      </main>
      <FooterMinimal />
    </>
  );
}
