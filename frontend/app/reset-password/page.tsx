"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { resetPassword, checkPasswordStrength } from "@/lib/auth";

function ResetPasswordForm() {
  const router       = useRouter();
  const searchParams = useSearchParams();
  const token        = searchParams.get("token") || "";

  const [password, setPassword] = useState("");
  const [confirm, setConfirm]   = useState("");
  const [error, setError]       = useState<string | null>(null);
  const [success, setSuccess]   = useState(false);
  const [loading, setLoading]   = useState(false);

  const strength = checkPasswordStrength(password);
  const allGood  = strength.length && strength.uppercase && strength.digit;

  useEffect(() => {
    if (!token) setError("Lien de réinitialisation manquant ou invalide.");
  }, [token]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password !== confirm) { setError("Les mots de passe ne correspondent pas"); return; }
    if (!allGood) { setError("Le mot de passe ne respecte pas les critères"); return; }
    setError(null);
    setLoading(true);
    try {
      await resetPassword(token, password);
      setSuccess(true);
      setTimeout(() => router.push("/login"), 2500);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur");
    } finally {
      setLoading(false);
    }
  };

  return (
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
          Nouveau mot de passe
        </h1>
      </div>

      {success ? (
        <div style={{
          padding: "24px", borderRadius: "16px", textAlign: "center",
          background: "rgba(126,201,160,0.06)", border: "1px solid rgba(126,201,160,0.2)",
        }}>
          <p style={{ fontSize: "14px", color: "#7ec9a0", marginBottom: "8px" }}>
            Mot de passe modifié
          </p>
          <p style={{ fontSize: "13px", color: "var(--muted)" }}>
            Redirection vers la connexion…
          </p>
        </div>
      ) : (
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>

          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <label style={labelStyle}>Nouveau mot de passe</label>
            <input suppressHydrationWarning
              type="password" value={password} onChange={e => setPassword(e.target.value)}
              placeholder="••••••••" required autoFocus
              style={inputStyle}
            />
            {password.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: "4px", marginTop: "4px" }}>
                {[
                  { ok: strength.length,    label: "8 caractères minimum" },
                  { ok: strength.uppercase, label: "Une majuscule" },
                  { ok: strength.digit,     label: "Un chiffre" },
                ].map(r => (
                  <div key={r.label} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <div style={{
                      width: "6px", height: "6px", borderRadius: "50%",
                      background: r.ok ? "#7ec9a0" : "var(--border)", transition: "background 0.2s",
                    }} />
                    <span style={{ fontSize: "12px", color: r.ok ? "#7ec9a0" : "var(--muted)" }}>
                      {r.label}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <label style={labelStyle}>Confirmer</label>
            <input suppressHydrationWarning
              type="password" value={confirm} onChange={e => setConfirm(e.target.value)}
              placeholder="••••••••" required
              style={{
                ...inputStyle,
                borderColor: confirm.length > 0 ? (confirm === password ? "rgba(126,201,160,0.4)" : "rgba(232,112,112,0.4)") : "var(--border)",
              }}
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

          <button type="submit" disabled={loading || !token} style={{
            marginTop: "8px", padding: "14px", borderRadius: "14px",
            cursor: loading ? "wait" : "pointer",
            fontSize: "14px", fontWeight: 500, border: "none",
            background: loading ? "var(--border)" : "linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%)",
            color: loading ? "var(--muted)" : "#0c0c0e",
            transition: "all 0.2s",
          }}>
            {loading ? "Enregistrement…" : "Enregistrer le nouveau mot de passe"}
          </button>

          <Link href="/login" style={{ textAlign: "center", fontSize: "13px", color: "var(--muted)", textDecoration: "none" }}>
            ← Retour à la connexion
          </Link>
        </form>
      )}
    </div>
  );
}

export default function ResetPasswordPage() {
  return (
    <main style={{
      minHeight: "100vh", display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      padding: "24px", background: "var(--background)",
    }}>
      <Suspense fallback={<p style={{ color: "var(--muted)" }}>Chargement…</p>}>
        <ResetPasswordForm />
      </Suspense>
      <Footer />
    </main>
  );
}

const labelStyle: React.CSSProperties = {
  fontSize: "11px", letterSpacing: "0.15em", textTransform: "uppercase", color: "var(--muted)",
};
const inputStyle: React.CSSProperties = {
  padding: "12px 16px", borderRadius: "12px", fontSize: "14px",
  background: "var(--surface)", border: "1px solid var(--border)",
  color: "var(--foreground)", outline: "none", width: "100%", transition: "border-color 0.2s",
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
