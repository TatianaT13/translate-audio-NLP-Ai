"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { login } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");
  const [error, setError]       = useState<string | null>(null);
  const [loading, setLoading]   = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await login(email, password);
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur de connexion");
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
            Connexion
          </h1>
          <p style={{ fontSize: "13px", color: "var(--muted)", lineHeight: 1.6 }}>
            Pas encore de compte ?{" "}
            <Link href="/register" style={{ color: "var(--accent)", textDecoration: "none" }}>
              Créer un compte
            </Link>
          </p>
        </div>

        {/* Form */}
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

          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <label style={{ fontSize: "11px", letterSpacing: "0.15em", textTransform: "uppercase", color: "var(--muted)" }}>
                Mot de passe
              </label>
              <Link href="/forgot-password" style={{ fontSize: "12px", color: "var(--accent)", opacity: 0.7, textDecoration: "none" }}>
                Mot de passe oublié ?
              </Link>
            </div>
            <input
              type="password" value={password} onChange={e => setPassword(e.target.value)}
              placeholder="••••••••" required
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
            marginTop: "8px", padding: "14px", borderRadius: "14px", cursor: loading ? "wait" : "pointer",
            fontSize: "14px", fontWeight: 500, border: "none",
            background: loading ? "var(--border)" : "linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%)",
            color: loading ? "var(--muted)" : "#0c0c0e",
            transition: "all 0.2s",
          }}>
            {loading ? "Connexion…" : "Se connecter"}
          </button>
        </form>
      </div>

      <Footer />
    </main>
  );
}

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
