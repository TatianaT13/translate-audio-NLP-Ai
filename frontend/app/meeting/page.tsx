"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { getMe, logout } from "@/lib/auth";
import type { User } from "@/lib/auth";
import { transcribeChunk, summarizeMeeting, type ChunkTranscript, type SummaryResponse } from "@/lib/meeting";

type Step = "idle" | "recording" | "processing_chunk" | "stopped" | "summarizing" | "summarized" | "error";

const SUMMARY_LANGS = [
  { code: "fr", label: "Français" },
  { code: "en", label: "Anglais"  },
  { code: "uk", label: "Ukrainien" },
  { code: "es", label: "Espagnol" },
  { code: "de", label: "Allemand" },
];

const STYLES = [
  { value: "executive",     label: "Synthèse exécutive (rapide)" },
  { value: "detailed",      label: "Compte-rendu détaillé"        },
  { value: "actions_only",  label: "Liste d'actions uniquement"   },
];

const CHUNK_DURATION_MS = 30_000;   // 30s par chunk

interface TranscriptSegment {
  text:       string;
  language:   string;
  confidence: number;
  ts:         string;
}

export default function MeetingPage() {
  const router = useRouter();

  const [user,       setUser]       = useState<User | null>(null);
  const [step,       setStep]       = useState<Step>("idle");
  const [segments,   setSegments]   = useState<TranscriptSegment[]>([]);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [error,      setError]      = useState<string | null>(null);

  const [summary,     setSummary]     = useState<SummaryResponse | null>(null);
  const [summaryLang, setSummaryLang] = useState("fr");
  const [summaryStyle, setSummaryStyle] = useState<"executive" | "detailed" | "actions_only">("executive");

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef        = useRef<MediaStream | null>(null);
  const chunkTimerRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const elapsedTimerRef  = useRef<ReturnType<typeof setInterval> | null>(null);

  // Auth check
  useEffect(() => {
    getMe().then(setUser).catch(() => router.push("/login"));
  }, [router]);

  // ── Recording lifecycle ────────────────────────────────────────────────
  const startRecording = useCallback(async () => {
    setError(null);
    setSegments([]);
    setSummary(null);
    setElapsedSec(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mr = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
      mediaRecorderRef.current = mr;

      // À chaque dataavailable, on envoie le chunk au STT (en background)
      mr.ondataavailable = async (e) => {
        if (e.data.size < 1000) return;       // ignore les chunks vides
        setStep("processing_chunk");
        try {
          const t = await transcribeChunk(e.data);
          if (t.text.trim().length > 0) {
            setSegments(prev => [...prev, {
              text:       t.text,
              language:   t.language,
              confidence: t.confidence,
              ts:         new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
            }]);
          }
        } catch (err) {
          console.error("STT chunk error:", err);
        } finally {
          setStep("recording");
        }
      };

      // Démarrer avec timeslice → flush automatique toutes les 30s
      mr.start(CHUNK_DURATION_MS);
      setStep("recording");

      // Timer affichage
      elapsedTimerRef.current = setInterval(() => setElapsedSec(s => s + 1), 1000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Microphone inaccessible.");
      setStep("error");
    }
  }, []);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
    streamRef.current?.getTracks().forEach(t => t.stop());
    if (chunkTimerRef.current)    clearInterval(chunkTimerRef.current);
    if (elapsedTimerRef.current)  clearInterval(elapsedTimerRef.current);
    setStep("stopped");
  }, []);

  // Cleanup au unmount
  useEffect(() => {
    return () => {
      mediaRecorderRef.current?.stop();
      streamRef.current?.getTracks().forEach(t => t.stop());
      if (chunkTimerRef.current)   clearInterval(chunkTimerRef.current);
      if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
    };
  }, []);

  // ── Summary ───────────────────────────────────────────────────────────
  const generateSummary = useCallback(async () => {
    const transcript = segments.map(s => s.text).join(" ");
    if (!transcript) {
      setError("Aucun transcript à résumer.");
      return;
    }
    setStep("summarizing");
    setError(null);
    try {
      const res = await summarizeMeeting(transcript, summaryLang, summaryStyle);
      setSummary(res);
      setStep("summarized");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur résumé");
      setStep("error");
    }
  }, [segments, summaryLang, summaryStyle]);

  const downloadMarkdown = useCallback(() => {
    if (!summary) return;
    const blob = new Blob([summary.summary], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `meeting_${Date.now()}_${summary.target_lang}.md`;
    a.click();
  }, [summary]);

  const downloadTranscript = useCallback(() => {
    const content = segments.map(s => `[${s.ts}] (${s.language}) ${s.text}`).join("\n\n");
    const blob = new Blob([content], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `transcript_${Date.now()}.txt`;
    a.click();
  }, [segments]);

  // ── UI ────────────────────────────────────────────────────────────────
  const mmss = `${Math.floor(elapsedSec / 60).toString().padStart(2,"0")}:${(elapsedSec % 60).toString().padStart(2,"0")}`;
  const transcriptText = segments.map(s => s.text).join(" ");

  if (!user) return null;

  return (
    <main style={{
      minHeight: "100vh", background: "var(--background)",
      padding: "32px 24px 80px",
      display: "flex", flexDirection: "column", alignItems: "center",
    }}>
      <div style={{ width: "100%", maxWidth: "720px" }}>

        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "24px" }}>
          <div>
            <div style={{
              display: "inline-block", fontSize: "10px", letterSpacing: "0.35em",
              textTransform: "uppercase", marginBottom: "10px",
              padding: "4px 12px", borderRadius: "999px",
              background: "rgba(201,169,110,0.08)", color: "var(--accent)",
            }}>
              Meeting Recorder
            </div>
            <h1 className="font-serif" style={{ fontSize: "clamp(24px, 4vw, 36px)", color: "var(--foreground)", lineHeight: 1.2 }}>
              Compte-rendu de <em style={{ color: "var(--accent)" }}>réunion</em>
            </h1>
            <p style={{ fontSize: "13px", color: "var(--muted)", marginTop: "6px" }}>
              Enregistrement en continu · transcription par chunks de 30s · résumé multi-langue
            </p>
          </div>
          <button onClick={() => router.push("/")} style={{
            padding: "7px 16px", borderRadius: "999px", fontSize: "12px",
            cursor: "pointer", background: "var(--surface)", border: "1px solid var(--border)", color: "var(--muted)",
          }}>
            ← Traduction
          </button>
        </div>

        {/* Recording controls */}
        <div style={{
          padding: "20px 24px", borderRadius: "16px",
          background: "var(--surface)", border: "1px solid var(--border)",
          marginBottom: "24px",
        }}>
          {step === "idle" && (
            <div style={{ textAlign: "center" }}>
              <button onClick={startRecording} style={{
                padding: "14px 32px", borderRadius: "999px", cursor: "pointer",
                fontSize: "14px", fontWeight: 600,
                background: "linear-gradient(135deg, var(--accent), var(--accent-dim))",
                color: "#0c0c0e", border: "none",
              }}>
                ● Démarrer l&apos;enregistrement
              </button>
              <p style={{ fontSize: "11px", color: "var(--muted)", marginTop: "10px" }}>
                Autorisez l&apos;accès au micro · les chunks sont transcrits en temps réel
              </p>
            </div>
          )}

          {(step === "recording" || step === "processing_chunk") && (
            <div style={{ display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap" }}>
              <div style={{
                width: "12px", height: "12px", borderRadius: "50%",
                background: "#e87070", animation: "pulse-dot 1s infinite",
              }} />
              <span style={{ fontSize: "16px", fontWeight: 600, color: "var(--foreground)", fontVariantNumeric: "tabular-nums" }}>
                {mmss}
              </span>
              <span style={{ fontSize: "11px", color: "var(--muted)", marginLeft: "auto" }}>
                {step === "processing_chunk" ? "Transcription du chunk…" : `${segments.length} segment${segments.length > 1 ? "s" : ""}`}
              </span>
              <button onClick={stopRecording} style={{
                padding: "10px 22px", borderRadius: "999px", cursor: "pointer",
                fontSize: "13px", fontWeight: 500,
                background: "var(--surface)", color: "var(--foreground)",
                border: "1px solid var(--border)",
              }}>
                ⏹ Arrêter
              </button>
            </div>
          )}

          {step === "stopped" && (
            <div style={{ textAlign: "center" }}>
              <p style={{ fontSize: "13px", color: "var(--foreground)", marginBottom: "16px" }}>
                Enregistrement terminé · {segments.length} segment{segments.length > 1 ? "s" : ""} · durée {mmss}
              </p>
              <div style={{ display: "flex", gap: "10px", justifyContent: "center", flexWrap: "wrap" }}>
                <button onClick={startRecording} style={{
                  padding: "10px 22px", borderRadius: "999px", cursor: "pointer",
                  fontSize: "13px", background: "var(--surface)", border: "1px solid var(--border)", color: "var(--muted)",
                }}>
                  ↻ Nouveau
                </button>
                <button onClick={downloadTranscript} style={{
                  padding: "10px 22px", borderRadius: "999px", cursor: "pointer",
                  fontSize: "13px", background: "var(--surface)", border: "1px solid var(--border)", color: "var(--foreground)",
                }}>
                  ⬇ Transcript .txt
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Live transcript */}
        {segments.length > 0 && (
          <div style={{ marginBottom: "24px" }}>
            <h2 style={{ fontSize: "11px", letterSpacing: "0.25em", textTransform: "uppercase", color: "var(--muted)", marginBottom: "10px", fontWeight: 600 }}>
              Transcript en direct
            </h2>
            <div style={{
              padding: "18px 22px", borderRadius: "14px",
              background: "var(--surface)", border: "1px solid var(--border)",
              maxHeight: "280px", overflowY: "auto",
              display: "flex", flexDirection: "column", gap: "10px",
            }}>
              {segments.map((s, i) => (
                <div key={i} style={{ fontSize: "13px", lineHeight: 1.6, color: "var(--foreground)" }}>
                  <span style={{ fontSize: "10px", color: "var(--muted)", marginRight: "8px", fontFamily: "monospace" }}>
                    [{s.ts}]
                  </span>
                  <span style={{
                    fontSize: "9px", padding: "1px 6px", borderRadius: "999px",
                    background: "rgba(201,169,110,0.1)", color: "var(--accent)", marginRight: "8px",
                  }}>
                    {s.language.toUpperCase()}
                  </span>
                  {s.text}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Summary controls */}
        {(step === "stopped" || step === "summarizing" || step === "summarized") && transcriptText.length > 0 && (
          <div style={{
            padding: "20px 24px", borderRadius: "16px",
            background: "var(--surface)", border: "1px solid var(--accent-dim)",
            marginBottom: "24px",
          }}>
            <h2 style={{ fontSize: "11px", letterSpacing: "0.25em", textTransform: "uppercase", color: "var(--accent)", marginBottom: "12px", fontWeight: 600 }}>
              Générer un compte-rendu
            </h2>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "14px" }}>
              <div>
                <label style={{ fontSize: "10px", color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase" }}>
                  Langue
                </label>
                <select value={summaryLang} onChange={e => setSummaryLang(e.target.value)} style={{
                  width: "100%", padding: "8px 10px", marginTop: "4px",
                  borderRadius: "8px", border: "1px solid var(--border)",
                  background: "var(--background)", color: "var(--foreground)", fontSize: "12px",
                }}>
                  {SUMMARY_LANGS.map(l => <option key={l.code} value={l.code}>{l.label}</option>)}
                </select>
              </div>
              <div>
                <label style={{ fontSize: "10px", color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase" }}>
                  Style
                </label>
                <select value={summaryStyle} onChange={e => setSummaryStyle(e.target.value as "executive" | "detailed" | "actions_only")} style={{
                  width: "100%", padding: "8px 10px", marginTop: "4px",
                  borderRadius: "8px", border: "1px solid var(--border)",
                  background: "var(--background)", color: "var(--foreground)", fontSize: "12px",
                }}>
                  {STYLES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                </select>
              </div>
            </div>

            <button onClick={generateSummary} disabled={step === "summarizing"} style={{
              width: "100%", padding: "12px", borderRadius: "10px", cursor: step === "summarizing" ? "wait" : "pointer",
              fontSize: "13px", fontWeight: 600,
              background: step === "summarizing"
                ? "var(--border)"
                : "linear-gradient(135deg, var(--accent), var(--accent-dim))",
              color: step === "summarizing" ? "var(--muted)" : "#0c0c0e",
              border: "none",
            }}>
              {step === "summarizing" ? "Génération en cours…" : "✨ Générer le compte-rendu"}
            </button>
          </div>
        )}

        {/* Summary result */}
        {summary && (
          <div style={{
            padding: "20px 24px", borderRadius: "16px",
            background: "rgba(201,169,110,0.04)", border: "1px solid var(--accent-dim)",
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px", flexWrap: "wrap", gap: "8px" }}>
              <h2 style={{ fontSize: "11px", letterSpacing: "0.25em", textTransform: "uppercase", color: "var(--accent)", fontWeight: 600 }}>
                Compte-rendu · {SUMMARY_LANGS.find(l => l.code === summary.target_lang)?.label} · {summary.style}
              </h2>
              <button onClick={downloadMarkdown} style={{
                padding: "5px 12px", borderRadius: "8px", fontSize: "11px", cursor: "pointer",
                background: "rgba(201,169,110,0.1)", border: "1px solid var(--accent-dim)", color: "var(--accent)",
              }}>
                ⬇ .md
              </button>
            </div>
            <pre style={{
              whiteSpace: "pre-wrap", fontFamily: "var(--font-playfair), Georgia, serif",
              fontSize: "14px", lineHeight: 1.7, color: "var(--foreground)", margin: 0,
            }}>
              {summary.summary}
            </pre>
            {summary.cost_usd != null && (
              <p style={{ fontSize: "10px", color: "var(--muted)", marginTop: "14px", textAlign: "right" }}>
                {summary.total_tokens} tokens · ${summary.cost_usd.toFixed(5)} · {summary.latency_ms}ms
              </p>
            )}
          </div>
        )}

        {error && (
          <div style={{
            padding: "12px 16px", borderRadius: "10px", fontSize: "12px",
            background: "rgba(232,112,112,0.08)", border: "1px solid rgba(232,112,112,0.2)", color: "#e87070",
            marginTop: "16px",
          }}>
            {error}
          </div>
        )}

        {/* Footer link back */}
        <p style={{ fontSize: "11px", color: "var(--muted)", opacity: 0.6, textAlign: "center", marginTop: "32px" }}>
          ✨ Powered by Whisper + Llama (Groq) · Multi-langue · MVP
        </p>
      </div>

      <style>{`
        @keyframes pulse-dot {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.4; transform: scale(0.85); }
        }
      `}</style>
    </main>
  );
}
