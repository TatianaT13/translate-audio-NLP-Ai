"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { register, checkPasswordStrength } from "@/lib/auth";

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm]   = useState("");
  const [error, setError]       = useState<string | null>(null);
  const [loading, setLoading]   = useState(false);

  const strength = checkPasswordStrength(password);
  const allGood  = strength.length && strength.uppercase && strength.digit;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (password !== confirm) { setError("Les mots de passe ne correspondent pas"); return; }
    if (!allGood) { setError("Le mot de passe ne respecte pas les critères"); return; }
    setLoading(true);
    try {
      await register(email, password);
      router.push("/login?registered=1");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur lors de la création du compte");
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

        {/* Header */}
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
            fontSize: "clamp(28px, 5vw, 38px)", color: "var(--foreground)",
            lineHeight: 1.15, marginBottom: "10px",
          }}>
            Créer un compte
          </h1>
          <p style={{ fontSize: "13px", color: "var(--muted)", lineHeight: 1.6 }}>
            Déjà inscrit ?{" "}
            <Link href="/login" style={{ color: "var(--accent)", textDecoration: "none" }}>
              Se connecter
            </Link>
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>

          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <label style={labelStyle}>Email</label>
            <input
              type="email" value={email} onChange={e => setEmail(e.target.value)}
              placeholder="vous@exemple.com" required autoFocus
              style={inputStyle}
            />
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <label style={labelStyle}>Mot de passe</label>
            <input
              type="password" value={password} onChange={e => setPassword(e.target.value)}
              placeholder="••••••••" required
              style={inputStyle}
            />
            {/* Strength indicators */}
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
                      background: r.ok ? "#7ec9a0" : "var(--border)",
                      transition: "background 0.2s",
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
            <label style={labelStyle}>Confirmer le mot de passe</label>
            <input
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

          <button type="submit" disabled={loading} style={{
            marginTop: "8px", padding: "14px", borderRadius: "14px",
            cursor: loading ? "wait" : "pointer",
            fontSize: "14px", fontWeight: 500, border: "none",
            background: loading ? "var(--border)" : "linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%)",
            color: loading ? "var(--muted)" : "#0c0c0e",
            transition: "all 0.2s",
          }}>
            {loading ? "Création…" : "Créer mon compte"}
          </button>
        </form>
      </div>

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
  color: "var(--foreground)", outline: "none", width: "100%",
  transition: "border-color 0.2s",
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
