"use client";

import { Suspense, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { AppHeader } from "@/components/AppHeader";
import { FooterMinimal } from "@/components/Footer";

function SuccesContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("session_id");

  useEffect(() => {
    // Redirect auto vers /translate apres 5s si l'user ne clique rien
    const t = setTimeout(() => router.push("/translate"), 5000);
    return () => clearTimeout(t);
  }, [router]);

  return (
    <main style={{
      minHeight: "calc(100vh - 65px)",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      padding: "24px", background: "var(--background)",
    }}>
      <div style={{ maxWidth: "480px", textAlign: "center" }}>

        {/* Icon check */}
        <div style={{
          width: "64px", height: "64px", borderRadius: "50%",
          background: "rgba(126,201,160,0.10)",
          border: "1px solid rgba(126,201,160,0.4)",
          display: "flex", alignItems: "center", justifyContent: "center",
          margin: "0 auto 24px",
        }}>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#7ec9a0" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </div>

        <h1 style={{
          fontFamily: "var(--font-display), serif",
          fontSize: "2rem", fontWeight: 400,
          color: "var(--foreground)", margin: "0 0 12px 0",
          letterSpacing: "-0.015em",
        }}>
          Merci pour votre abonnement<span style={{ color: "var(--accent)" }}>.</span>
        </h1>

        <p style={{
          fontSize: "15px", lineHeight: 1.6,
          color: "var(--muted)", margin: "0 0 32px 0",
        }}>
          Votre paiement a été confirmé. Votre compte est maintenant en formule Pro,
          vous pouvez profiter de toutes les fonctionnalités dès maintenant.
        </p>

        <div style={{ display: "flex", gap: "12px", justifyContent: "center", flexWrap: "wrap" }}>
          <Link href="/translate" style={{
            padding: "12px 24px", borderRadius: "12px",
            background: "linear-gradient(135deg, var(--accent), var(--accent-dim))",
            color: "#0c0c0e", fontSize: "14px", fontWeight: 500,
            textDecoration: "none", letterSpacing: "0.02em",
          }}>Commencer à traduire</Link>
          <Link href="/" style={{
            padding: "12px 24px", borderRadius: "12px",
            background: "var(--surface)", border: "1px solid var(--border)",
            color: "var(--muted)", fontSize: "14px",
            textDecoration: "none",
          }}>Retour à l&apos;accueil</Link>
        </div>

        {sessionId && (
          <p style={{
            fontSize: "10px", color: "var(--muted)", opacity: 0.5,
            marginTop: "32px", fontFamily: "ui-monospace, monospace",
          }}>
            Session : {sessionId.slice(0, 24)}…
          </p>
        )}

        <p style={{
          fontSize: "11px", color: "var(--muted)", opacity: 0.6,
          marginTop: "24px",
        }}>
          Vous allez être redirigé automatiquement dans quelques secondes.
        </p>
      </div>
    </main>
  );
}

export default function SuccesPage() {
  return (
    <>
      <AppHeader variant="landing" />
      <Suspense fallback={<div style={{ padding: "48px", textAlign: "center", color: "var(--muted)" }}>Chargement…</div>}>
        <SuccesContent />
      </Suspense>
      <FooterMinimal />
    </>
  );
}
