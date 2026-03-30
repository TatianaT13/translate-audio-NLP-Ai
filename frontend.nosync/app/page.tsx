"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { runPipeline, audioFromBase64 } from "@/lib/api";
import type { ProcessResult } from "@/lib/api";

type Step = "idle" | "recording" | "processing" | "done" | "error";

const LANGS = [
  { code: "en", label: "Anglais" },
  { code: "uk", label: "Ukrainien" },
  { code: "es", label: "Espagnol" },
  { code: "de", label: "Allemand" },
];

const S = {
  gap4:  "4px",
  gap8:  "8px",
  gap12: "12px",
  gap16: "16px",
  gap24: "24px",
  gap32: "32px",
  gap48: "48px",
};

const QUOTES = [
  { fr: "On ne voit bien qu'avec le cœur.",                  author: "Antoine de Saint-Exupéry", text: "One sees clearly only with the heart.",     lang: "EN", color: "#7eb8c9" },
  { fr: "Je pense, donc je suis.",                           author: "René Descartes",            text: "Я мислю, отже я існую.",                   lang: "UK", color: "#9b7ec9" },
  { fr: "La vie est courte, l'art est long.",                author: "Hippocrate",                text: "La vida es corta, el arte es largo.",      lang: "ES", color: "#c9a96e" },
  { fr: "L'union fait la force.",                            author: "Proverbe belge",            text: "Einigkeit macht stark.",                   lang: "DE", color: "#7ec9a0" },

  { fr: "Connais-toi toi-même.",                             author: "Socrate",                   text: "Know thyself.",                            lang: "EN", color: "#7eb8c9" },
  { fr: "Le silence est d'or.",                              author: "Proverbe français",          text: "Мовчання — золото.",                       lang: "UK", color: "#9b7ec9" },
  { fr: "Rien ne se perd, rien ne se crée.",                 author: "Antoine Lavoisier",         text: "Nada se pierde, todo se transforma.",      lang: "ES", color: "#c9a96e" },
  { fr: "Le doute est le commencement de la sagesse.",       author: "Aristote",                  text: "Der Zweifel ist der Beginn der Weisheit.", lang: "DE", color: "#7ec9a0" },

  { fr: "Le temps, c'est de l'argent.",                      author: "Proverbe",                  text: "Time is money.",                           lang: "EN", color: "#7eb8c9" },
  { fr: "Les mots sont les fenêtres de l'âme.",              author: "Proverbe",                  text: "Слова — це вікна душі.",                   lang: "UK", color: "#9b7ec9" },
  { fr: "Mieux vaut tard que jamais.",                       author: "Proverbe",                  text: "Más vale tarde que nunca.",                lang: "ES", color: "#c9a96e" },
  { fr: "L'erreur est humaine.",                             author: "Proverbe",                  text: "Irren ist menschlich.",                    lang: "DE", color: "#7ec9a0" },

  { fr: "La nuit porte conseil.",                            author: "Proverbe français",          text: "Sleep on it — the night brings counsel.",  lang: "EN", color: "#7eb8c9" },
  { fr: "La beauté est dans les yeux de celui qui regarde.", author: "Proverbe",                  text: "Краса в очах того, хто дивиться.",         lang: "UK", color: "#9b7ec9" },
  { fr: "Vouloir, c'est pouvoir.",                           author: "Proverbe",                  text: "Querer es poder.",                         lang: "ES", color: "#c9a96e" },
  { fr: "Il n'est jamais trop tard pour bien faire.",        author: "Proverbe",                  text: "Es ist nie zu spät, Gutes zu tun.",        lang: "DE", color: "#7ec9a0" },
];

function MiniWave({ color }: { color: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "2px", margin: "0 18px", flexShrink: 0 }}>
      {[6, 11, 8, 14, 9, 14, 8, 11, 6].map((h, i) => (
        <span key={i} className="wave-bar" style={{
          display: "inline-block", width: "2px", height: `${h}px`, borderRadius: "1px",
          background: color, opacity: 0.55, animationDelay: `${i * 0.09}s`,
        }} />
      ))}
    </span>
  );
}

function WaveTransform() {
  const items = [...QUOTES, ...QUOTES]; // duplicate for seamless loop

  return (
    <div style={{
      width: "100%", overflow: "hidden", marginBottom: S.gap24,
      WebkitMaskImage: "linear-gradient(to right, transparent, black 64px, black calc(100% - 64px), transparent)",
      maskImage:       "linear-gradient(to right, transparent, black 64px, black calc(100% - 64px), transparent)",
    }}>
      <div style={{
        display: "inline-flex", alignItems: "center", whiteSpace: "nowrap",
        animation: "marquee 160s linear infinite",
      }}>
        {items.map((q, i) => (
          <span key={i} style={{ display: "inline-flex", alignItems: "center" }}>
            <span style={{
              fontSize: "19px", fontFamily: "var(--font-playfair), serif",
              fontStyle: "italic", color: "var(--foreground)", opacity: 0.78,
            }}>
              &ldquo;{q.fr}&rdquo;
            </span>

            <MiniWave color={q.color} />

            <span style={{
              fontSize: "19px", fontFamily: "var(--font-playfair), serif",
              fontStyle: "italic", color: q.color,
            }}>
              &ldquo;{q.text}&rdquo;
            </span>

            <span style={{
              fontSize: "10px", fontWeight: 600, letterSpacing: "0.18em",
              padding: "2px 8px", borderRadius: "999px", marginLeft: "10px",
              background: `${q.color}18`, color: q.color,
            }}>
              {q.lang}
            </span>

            <span style={{
              fontSize: "11px", color: "var(--muted)", opacity: 0.65,
              margin: "0 48px 0 12px", letterSpacing: "0.1em",
            }}>
              — {q.author}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}

function Toast({ message, onDone }: { message: string; onDone: () => void }) {
  useEffect(() => {
    const t = setTimeout(onDone, 2000);
    return () => clearTimeout(t);
  }, [onDone]);
  return (
    <div style={{
      position: "fixed", bottom: "32px", left: "50%", transform: "translateX(-50%)",
      padding: "12px 24px", borderRadius: "999px", fontSize: "13px", fontWeight: 500,
      background: "var(--accent)", color: "#0c0c0e",
      boxShadow: "0 8px 32px rgba(201,169,110,0.3)",
      animation: "fadeUp 0.3s ease forwards", zIndex: 50,
    }}>
      {message}
    </div>
  );
}

export default function Home() {
  const [step, setStep]         = useState<Step>("idle");
  const [result, setResult]     = useState<ProcessResult | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError]       = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [targetLang, setTargetLang] = useState("en");
  const [toast, setToast]       = useState<string | null>(null);
  const [copied, setCopied]     = useState(false);

  const fileRef      = useRef<HTMLInputElement>(null);
  const mediaRef     = useRef<MediaRecorder | null>(null);
  const chunksRef    = useRef<Blob[]>([]);
  const audioBlobRef = useRef<Blob | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.target !== document.body) return;
      e.preventDefault();
      if (step === "idle") startRecording();
      else if (step === "recording") stopRecording();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step]);

  const run = useCallback(async (file: File | Blob) => {
    setError(null); setResult(null); setAudioUrl(null); setCopied(false);

    const MAX_MB = 25;
    if (file.size > MAX_MB * 1024 * 1024) {
      setError(`Fichier trop volumineux (${(file.size / 1024 / 1024).toFixed(1)} Mo). Maximum : ${MAX_MB} Mo (~20 min d'audio).`);
      setStep("error");
      return;
    }

    setStep("processing");
    try {
      const res = await runPipeline(file, targetLang);
      setResult(res);
      if (res.audio_b64) {
        const blob = audioFromBase64(res.audio_b64, res.audio_content_type);
        audioBlobRef.current = blob;
        setAudioUrl(URL.createObjectURL(blob));
      }
      setStep("done");
      setToast(res.language_prob < 0.7 ? "Confiance faible — vérifiez la transcription" : "Traduction terminée");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStep("error");
    }
  }, [targetLang]);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) run(f);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        stream.getTracks().forEach(t => t.stop());
        run(blob);
      };
      mr.start(); mediaRef.current = mr; setStep("recording");
    } catch {
      setError("Microphone inaccessible."); setStep("error");
    }
  };

  const stopRecording = () => mediaRef.current?.stop();

  const copyTranslation = () => {
    if (!result) return;
    navigator.clipboard.writeText(result.translation);
    setCopied(true); setTimeout(() => setCopied(false), 2000);
  };

  const downloadAudio = () => {
    if (!audioBlobRef.current) return;
    const ext = audioBlobRef.current.type.includes("mpeg") ? "mp3" : "wav";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(audioBlobRef.current);
    a.download = `translation.${ext}`; a.click();
  };

  const reset = () => { setStep("idle"); setResult(null); setAudioUrl(null); setError(null); setCopied(false); };

  const langLabel = LANGS.find(l => l.code === targetLang)?.label ?? "English";

  return (
    <main style={{
      minHeight: "100vh", display: "flex", flexDirection: "column",
      alignItems: "center", padding: "40px 24px 72px",
      background: "var(--background)",
    }}>
      {toast && <Toast message={toast} onDone={() => setToast(null)} />}

      {/* ── Header ── */}
      <header style={{
        textAlign: "center", width: "100%", maxWidth: "680px",
        marginBottom: step === "done" ? S.gap24 : S.gap32,
        animation: "fadeUp 0.5s ease forwards",
      }}>
        <div style={{
          display: "inline-block", fontSize: "11px", letterSpacing: "0.35em",
          textTransform: "uppercase", marginBottom: S.gap24,
          padding: "6px 16px", borderRadius: "999px",
          background: "rgba(201,169,110,0.08)", color: "var(--accent)",
        }}>
          Traduction Audio IA
        </div>

        {step === "done" ? (
          <h1 className="font-serif" style={{ fontSize: "clamp(22px, 4vw, 30px)", color: "var(--foreground)", lineHeight: 1.2 }}>
            Traduisez votre voix.{" "}
            <em style={{ color: "var(--accent)" }}>Instantanément.</em>
          </h1>
        ) : (
          <>
            <h1 className="font-serif" style={{
              fontSize: "clamp(32px, 6vw, 54px)", color: "var(--foreground)",
              lineHeight: 1.1, marginBottom: S.gap16,
            }}>
              Traduisez votre voix.
              <br />
              <em style={{ color: "var(--accent)" }}>Instantanément.</em>
            </h1>
            <p style={{
              fontSize: "13px", lineHeight: 1.6, fontWeight: 300,
              color: "var(--muted)", maxWidth: "42ch", margin: "0 auto",
            }}>
              Déposez ou enregistrez un audio — transcription, traduction et lecture en {langLabel}.
            </p>
          </>
        )}
      </header>

      <div style={{ width: "100%", maxWidth: "680px" }}>

        {/* ── Idle ── */}
        {step === "idle" && (
          <div style={{ animation: "fadeUp 0.5s ease forwards" }}>

            {/* Language selector */}
            <div style={{ display: "flex", gap: S.gap8, justifyContent: "center", flexWrap: "wrap", marginBottom: S.gap24 }}>
              {LANGS.map(l => (
                <button key={l.code} onClick={() => setTargetLang(l.code)} style={{
                  padding: "7px 18px", borderRadius: "999px", fontSize: "13px",
                  fontWeight: 500, cursor: "pointer", transition: "all 0.2s",
                  background: targetLang === l.code ? "rgba(201,169,110,0.12)" : "var(--surface)",
                  border: `1px solid ${targetLang === l.code ? "var(--accent)" : "var(--border)"}`,
                  color: targetLang === l.code ? "var(--accent)" : "var(--muted)",
                }}>
                  {l.label}
                </button>
              ))}
            </div>

            <WaveTransform />

            {/* Drop zone */}
            <div
              onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              onClick={() => fileRef.current?.click()}
              style={{
                borderRadius: "20px", textAlign: "center", cursor: "pointer",
                padding: "36px 24px", marginBottom: S.gap24,
                background: isDragging ? "rgba(201,169,110,0.05)" : "var(--surface)",
                border: `1.5px dashed ${isDragging ? "var(--accent)" : "var(--border)"}`,
                transition: "all 0.25s",
              }}
            >
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
                style={{ color: "var(--muted)", margin: "0 auto 20px" }}>
                <path d="M9 18V5l12-2v13" />
                <circle cx="6" cy="18" r="3" /><circle cx="18" cy="16" r="3" />
              </svg>
              <p style={{ fontSize: "15px", fontWeight: 400, color: "var(--foreground)", marginBottom: "6px" }}>
                Déposez un fichier audio ici
              </p>
              <p style={{ fontSize: "13px", color: "var(--muted)" }}>MP3, WAV, M4A, OGG</p>
              <input ref={fileRef} type="file" accept="audio/*" style={{ display: "none" }}
                onChange={e => e.target.files?.[0] && run(e.target.files[0])} />
            </div>

            {/* divider */}
            <div style={{ display: "flex", alignItems: "center", gap: S.gap16, marginBottom: S.gap32 }}>
              <div style={{ flex: 1, height: "1px", background: "var(--border)" }} />
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
                style={{ color: "var(--muted)", opacity: 0.4, flexShrink: 0 }}>
                <path d="M9 18V5l12-2v13" />
                <circle cx="6" cy="18" r="3" />
                <circle cx="18" cy="16" r="3" />
              </svg>
              <div style={{ flex: 1, height: "1px", background: "var(--border)" }} />
            </div>

            {/* Record button */}
            <button onClick={startRecording} style={{
              width: "100%", padding: "13px", borderRadius: "14px", cursor: "pointer",
              fontSize: "14px", fontWeight: 500, letterSpacing: "0.04em",
              background: "linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%)",
              color: "#0c0c0e", border: "none", transition: "opacity 0.2s",
            }}>
              Enregistrer avec le microphone
              <span style={{ marginLeft: "8px", fontSize: "12px", opacity: 0.6 }}>(ou appuyer sur Espace)</span>
            </button>
          </div>
        )}

        {/* ── Recording ── */}
        {step === "recording" && (
          <div style={{ textAlign: "center", padding: "48px 0", animation: "fadeUp 0.5s ease forwards" }}>
            <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "center", gap: "6px", height: "48px", marginBottom: "32px" }}>
              {[0.0, 0.1, 0.2, 0.3, 0.15, 0.25, 0.2].map((delay, i) => (
                <div key={i} className="wave-bar" style={{
                  width: "8px", height: "100%", borderRadius: "4px",
                  background: "var(--accent)", animationDelay: `${delay}s`,
                }} />
              ))}
            </div>
            <p style={{ fontSize: "14px", color: "var(--foreground)", marginBottom: "8px" }}>Enregistrement en cours…</p>
            <p style={{ fontSize: "12px", color: "var(--muted)", marginBottom: "32px" }}>Appuyez sur Espace ou cliquez pour arrêter</p>
            <button onClick={stopRecording} style={{
              padding: "12px 32px", borderRadius: "999px", fontSize: "13px",
              fontWeight: 500, cursor: "pointer", transition: "opacity 0.2s",
              background: "var(--surface)", border: "1px solid var(--border)", color: "var(--foreground)",
            }}>
              Arrêter et traduire
            </button>
          </div>
        )}

        {/* ── Processing ── */}
        {step === "processing" && (
          <div style={{ textAlign: "center", padding: "64px 0", animation: "fadeUp 0.5s ease forwards" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: S.gap12, marginBottom: S.gap24 }}>
              {["STT", "LLM", "TTS"].map((label, i) => (
                <div key={label} style={{ display: "flex", alignItems: "center", gap: S.gap12 }}>
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }}>
                    <div style={{
                      width: "10px", height: "10px", borderRadius: "50%",
                      background: "var(--accent)",
                      animation: `pulse 1.4s ease-in-out ${i * 0.3}s infinite`,
                    }} />
                    <span style={{ fontSize: "11px", letterSpacing: "0.1em", color: "var(--muted)" }}>{label}</span>
                  </div>
                  {i < 2 && <div style={{ width: "32px", height: "1px", background: "var(--border)", marginBottom: "18px" }} />}
                </div>
              ))}
            </div>
            <p style={{ fontSize: "13px", color: "var(--muted)" }}>Pipeline en cours…</p>
          </div>
        )}

        {/* ── Results ── */}
        {step === "done" && result && (
          <div style={{ display: "flex", flexDirection: "column", gap: S.gap32, animation: "fadeUp 0.5s ease forwards" }}>

            {/* Audio player */}
            {audioUrl && (
              <div style={{
                display: "flex", alignItems: "center", gap: S.gap16,
                padding: "16px 20px", borderRadius: "16px",
                background: "var(--surface)", border: "1px solid var(--accent-dim)",
              }}>
                <audio controls src={audioUrl} style={{ flex: 1, height: "36px", accentColor: "var(--accent)" }} />
                <button onClick={downloadAudio} title="Download" style={{
                  padding: "8px", borderRadius: "10px", cursor: "pointer",
                  background: "rgba(201,169,110,0.08)", border: "1px solid var(--accent-dim)",
                  color: "var(--accent)", flexShrink: 0, transition: "opacity 0.2s",
                }}>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
                  </svg>
                </button>
              </div>
            )}

            {/* Cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: S.gap24 }}>

              {/* Original */}
              <div style={{ borderRadius: "16px", overflow: "hidden", background: "var(--surface)", border: "1px solid var(--border)" }}>
                <div style={{
                  padding: "14px 20px", display: "flex", justifyContent: "space-between",
                  alignItems: "center", borderBottom: "1px solid var(--border)",
                }}>
                  <span style={{ fontSize: "11px", letterSpacing: "0.2em", textTransform: "uppercase", color: "var(--muted)", fontWeight: 500 }}>
                    Transcription
                  </span>
                  <div style={{ display: "flex", alignItems: "center", gap: S.gap8 }}>
                    <div style={{ height: "4px", width: "48px", borderRadius: "999px", overflow: "hidden", background: "var(--border)" }}>
                      <div style={{ height: "100%", borderRadius: "999px", background: "var(--accent-dim)", width: `${Math.round(result.language_prob * 100)}%` }} />
                    </div>
                    <span style={{ fontSize: "11px", padding: "3px 10px", borderRadius: "999px", background: "rgba(201,169,110,0.08)", color: "var(--accent)" }}>
                      {result.language.toUpperCase()} {Math.round(result.language_prob * 100)}%
                    </span>
                  </div>
                </div>
                <div style={{ padding: "16px 20px", maxHeight: "160px", overflowY: "auto" }}>
                  <p style={{ fontSize: "13px", lineHeight: 1.7, color: "var(--foreground)", fontWeight: 300 }}>
                    {result.source_text}
                  </p>
                </div>
              </div>

              {/* Translation */}
              <div style={{ borderRadius: "16px", overflow: "hidden", background: "var(--surface)", border: "1px solid var(--accent-dim)" }}>
                <div style={{
                  padding: "12px 20px", display: "flex", justifyContent: "space-between",
                  alignItems: "center", borderBottom: "1px solid rgba(201,169,110,0.15)",
                }}>
                  <span style={{ fontSize: "11px", letterSpacing: "0.2em", textTransform: "uppercase", color: "var(--accent)", fontWeight: 500 }}>
                    {langLabel}
                  </span>
                  <button onClick={copyTranslation} style={{
                    display: "flex", alignItems: "center", gap: "6px",
                    padding: "4px 12px", borderRadius: "8px", fontSize: "12px",
                    cursor: "pointer", transition: "all 0.2s",
                    background: copied ? "rgba(201,169,110,0.18)" : "rgba(201,169,110,0.06)",
                    border: "1px solid var(--accent-dim)", color: "var(--accent)",
                  }}>
                    {copied ? (
                      <><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copié</>
                    ) : (
                      <><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copier</>
                    )}
                  </button>
                </div>
                <div style={{ padding: "16px 20px", maxHeight: "160px", overflowY: "auto" }}>
                  <p style={{ fontSize: "13px", lineHeight: 1.7, color: "var(--foreground)", fontWeight: 300 }}>
                    {result.translation}
                  </p>
                </div>
              </div>
            </div>

            {/* Metrics + New */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: S.gap12 }}>
              <div style={{ display: "flex", gap: S.gap8, flexWrap: "wrap" }}>
                {[
                  { label: "STT", ms: result.latency_stt_ms, color: "#7eb8c9" },
                  { label: "LLM", ms: result.latency_llm_ms, color: "var(--accent)" },
                  { label: "TTS", ms: result.latency_tts_ms, color: "#9b7ec9" },
                  { label: "Total", ms: result.latency_total_ms, color: "var(--muted)" },
                ].map(m => {
                  const secs = m.ms / 1000;
                  const display = secs >= 60
                    ? `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`
                    : `${secs.toFixed(1)}s`;
                  return (
                    <div key={m.label} style={{
                      display: "flex", flexDirection: "column", alignItems: "center", gap: "4px",
                      padding: "8px 14px", borderRadius: "12px",
                      background: "var(--surface)", border: "1px solid var(--border)",
                    }}>
                      <span style={{ fontSize: "12px", fontWeight: 500, fontVariantNumeric: "tabular-nums", color: m.color }}>{display}</span>
                      <span style={{ fontSize: "11px", color: "var(--muted)" }}>{m.label}</span>
                    </div>
                  );
                })}
              </div>
              <button onClick={reset} style={{
                padding: "10px 20px", borderRadius: "12px", fontSize: "13px",
                cursor: "pointer", transition: "opacity 0.2s",
                background: "var(--surface)", border: "1px solid var(--border)", color: "var(--muted)",
              }}>
                Nouvelle traduction
              </button>
            </div>
          </div>
        )}

        {/* ── Error ── */}
        {step === "error" && error && (
          <div style={{ textAlign: "center", padding: "48px 0", animation: "fadeUp 0.5s ease forwards" }}>
            <div style={{
              width: "48px", height: "48px", borderRadius: "50%", margin: "0 auto 20px",
              display: "flex", alignItems: "center", justifyContent: "center",
              background: "rgba(232,112,112,0.1)", border: "1px solid rgba(232,112,112,0.2)",
            }}>
              <span style={{ color: "#e87070", fontSize: "18px" }}>×</span>
            </div>
            <p style={{ fontSize: "14px", color: "#e87070", marginBottom: "8px" }}>Une erreur est survenue</p>
            <p style={{ fontSize: "13px", color: "var(--muted)", marginBottom: "32px", maxWidth: "42ch", margin: "0 auto 32px" }}>{error}</p>
            <button onClick={reset} style={{
              padding: "10px 28px", borderRadius: "999px", fontSize: "13px",
              cursor: "pointer", background: "var(--surface)",
              border: "1px solid var(--border)", color: "var(--foreground)",
            }}>
              Réessayer
            </button>
          </div>
        )}
      </div>

      {/* ── Footer ── */}
      <footer style={{
        position: "fixed", bottom: 0, left: 0, right: 0,
        padding: "14px 24px",
        borderTop: "1px solid var(--border)",
        background: "var(--background)",
        textAlign: "center",
        zIndex: 10,
      }}>
        <p style={{ fontSize: "11px", letterSpacing: "0.12em", color: "var(--muted)", opacity: 0.4 }}>
          © {new Date().getFullYear()} traduction-audio.fr · Whisper · Llama · Voxtral
        </p>
      </footer>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.3; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1.2); }
        }
      `}</style>
    </main>
  );
}
