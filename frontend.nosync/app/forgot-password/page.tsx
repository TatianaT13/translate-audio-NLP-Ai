"use client";

import { useState } from "react";
import Link from "next/link";
import { forgotPassword } from "@/lib/auth";

export default function ForgotPasswordPage() {
  const [email, setEmail]     = useState("");
  const [sent, setSent]       = useState(false);
  const [resetUrl, setResetUrl] = useState<string | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const data = await forgotPassword(email);
      setSent(true);
      // DEV MODE: API returns the reset URL directly
      if (data.reset_url) setResetUrl(data.reset_url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{
      minHeight: "100vh", display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      padding: "24px", background: "var(--background)",
    }}>
      <div style={{ width: "100%", maxWidth: "400px" }}>

        <div style={{ textAlign: "center", marginBottom: "40px" }}>
          <div style={{
            display: "inline-block", fontSize: "11px", letterSpacing: "0.35em",
            textTransform: "uppercase", marginBottom: "20px",
            padding: "6px 16px", borderRadius: "999px",
            background: "rgba(201,169,110,0.08)", color: "var(--accent)",
          }}>
            Traduction Audio IA
          </div>
          <h1 className="font-serif" style={{
            fontSize: "clamp(26px, 5vw, 36px)", color: "var(--foreground)",
            lineHeight: 1.15, marginBottom: "10px",
          }}>
            Mot de passe oublié
          </h1>
          <p style={{ fontSize: "13px", color: "var(--muted)", lineHeight: 1.6 }}>
            Un lien de réinitialisation vous sera envoyé.
          </p>
        </div>

        {!sent ? (
          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <label style={{ fontSize: "11px", letterSpacing: "0.15em", textTransform: "uppercase", color: "var(--muted)" }}>
                Email
              </label>
              <input
                type="email" value={email} onChange={e => setEmail(e.target.value)}
                placeholder="vous@exemple.com" required autoFocus
                style={inputStyle}
              />
            </div>

            {error && (
              <div style={{
                padding: "12px 16px", borderRadius: "12px", fontSize: "13px",
                background: "rgba(232,112,112,0.08)", border: "1px solid rgba(232,112,112,0.2)",
                color: "#e87070",
              }}>
                {error}
              </div>
            )}

            <button type="submit" disabled={loading} style={{
              marginTop: "8px", padding: "14px", borderRadius: "14px",
              cursor: loading ? "wait" : "pointer",
              fontSize: "14px", fontWeight: 500, border: "none",
              background: loading ? "var(--border)" : "linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%)",
              color: loading ? "var(--muted)" : "#0c0c0e",
              transition: "all 0.2s",
            }}>
              {loading ? "Envoi…" : "Envoyer le lien"}
            </button>

            <Link href="/login" style={{ textAlign: "center", fontSize: "13px", color: "var(--muted)", textDecoration: "none" }}>
              ← Retour à la connexion
            </Link>
          </form>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            <div style={{
              padding: "20px", borderRadius: "16px", textAlign: "center",
              background: "rgba(126,201,160,0.06)", border: "1px solid rgba(126,201,160,0.2)",
            }}>
              <p style={{ fontSize: "14px", color: "#7ec9a0", marginBottom: "6px" }}>
                Lien envoyé
              </p>
              <p style={{ fontSize: "13px", color: "var(--muted)", lineHeight: 1.6 }}>
                Vérifiez votre boîte mail.
              </p>
            </div>

            {/* DEV MODE: show direct link */}
            {resetUrl && (
              <div style={{
                padding: "16px", borderRadius: "12px",
                background: "rgba(201,169,110,0.06)", border: "1px solid var(--accent-dim)",
              }}>
                <p style={{ fontSize: "11px", letterSpacing: "0.15em", textTransform: "uppercase", color: "var(--accent)", marginBottom: "8px" }}>
                  Mode développement
                </p>
                <a href={resetUrl} style={{ fontSize: "12px", color: "var(--accent)", wordBreak: "break-all" }}>
                  {resetUrl}
                </a>
              </div>
            )}

            <Link href="/login" style={{ textAlign: "center", fontSize: "13px", color: "var(--muted)", textDecoration: "none" }}>
              ← Retour à la connexion
            </Link>
          </div>
        )}
      </div>

      <Footer />
    </main>
  );
}

const inputStyle: React.CSSProperties = {
  padding: "12px 16px", borderRadius: "12px", fontSize: "14px",
  background: "var(--surface)", border: "1px solid var(--border)",
  color: "var(--foreground)", outline: "none", width: "100%",
};

function Footer() {
  return (
    <footer style={{ position: "fixed", bottom: 0, left: 0, right: 0, padding: "14px 24px", textAlign: "center", borderTop: "1px solid var(--border)", background: "var(--background)" }}>
      <p style={{ fontSize: "11px", letterSpacing: "0.12em", color: "var(--muted)", opacity: 0.4 }}>
        © {new Date().getFullYear()} traduction-audio.fr · Whisper · Llama · Voxtral
      </p>
    </footer>
  );
}
