"use client";

import { useEffect, useRef, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { verifyEmail } from "@/lib/auth";

type State = "checking" | "ok" | "error";

function VerifyEmail() {
  const token = useSearchParams().get("token") || "";
  const [state, setState]     = useState<State>("checking");
  const [message, setMessage] = useState("");
  // Un lien ne sert qu'une fois : React rejoue les effets en développement,
  // le second appel le trouverait déjà utilisé.
  const sent = useRef(false);

  useEffect(() => {
    if (sent.current) return;
    sent.current = true;
    if (!token) {
      setState("error");
      setMessage("Lien de confirmation manquant ou invalide.");
      return;
    }
    verifyEmail(token)
      .then((r) => { setState("ok"); setMessage(r.message); })
      .catch((e) => { setState("error"); setMessage(e instanceof Error ? e.message : "Lien invalide ou expiré"); });
  }, [token]);

  const color = state === "ok" ? "#7ec9a0" : state === "error" ? "#e87070" : "var(--muted)";

  return (
    <div style={{ width: "100%", maxWidth: "420px", textAlign: "center" }}>
      <h1 className="font-serif" style={{
        fontSize: "clamp(26px, 5vw, 34px)", color: "var(--foreground)", lineHeight: 1.2, marginBottom: "16px",
      }}>
        {state === "checking" ? "Confirmation en cours…" : state === "ok" ? "Adresse confirmée" : "Confirmation impossible"}
      </h1>
      <p style={{ fontSize: "14px", color, lineHeight: 1.6, marginBottom: "28px" }}>{message}</p>
      {state === "ok" && (
        <Link href="/login?next=/translate" style={linkStyle}>Se connecter et commencer</Link>
      )}
      {state === "error" && (
        <p style={{ fontSize: "13px", color: "var(--muted)", lineHeight: 1.6 }}>
          Connectez-vous pour demander un nouvel email de confirmation.{" "}
          <Link href="/login" style={{ color: "var(--accent)", textDecoration: "none" }}>Se connecter</Link>
        </p>
      )}
    </div>
  );
}

const linkStyle: React.CSSProperties = {
  display: "inline-block", padding: "14px 24px", borderRadius: "14px", fontSize: "14px", fontWeight: 500,
  background: "var(--accent)", color: "#0c0c0e", textDecoration: "none",
};

export default function VerifyEmailPage() {
  return (
    <main style={{
      minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center",
      padding: "24px", background: "var(--background)",
    }}>
      <Suspense>
        <VerifyEmail />
      </Suspense>
    </main>
  );
}
