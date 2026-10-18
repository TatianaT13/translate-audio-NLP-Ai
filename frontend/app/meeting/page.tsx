"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { getMe } from "@/lib/auth";
import type { User } from "@/lib/auth";
import {
  transcribeChunk, summarizeMeeting, transcribeDiarized, formatDiarizedTranscript,
  type ChunkTranscript, type SummaryResponse, type DiarizedTranscript, type DiarizedSegment,
} from "@/lib/meeting";
import { AppHeader } from "@/components/AppHeader";
import { FooterMinimal } from "@/components/Footer";

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

const SPEAKER_COLORS: Record<string, string> = {
  A: "#c9a96e", B: "#7eb8c9", C: "#9b7ec9", D: "#7ec9a0",
  E: "#e8b04a", F: "#e87070", G: "#cd93d0", H: "#8bc9bb",
};
function speakerColor(id: string): string {
  return SPEAKER_COLORS[id] || "var(--muted)";
}
function fmtTs(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

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

  // ── Diarisation (identification des intervenants) ────────────────────────
  const [diarize,          setDiarize]          = useState(false);
  const [diarizedResult,   setDiarizedResult]   = useState<DiarizedTranscript | null>(null);
  const [diarizing,        setDiarizing]        = useState(false);
  const [speakerNames,     setSpeakerNames]     = useState<Record<string, string>>({});

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef        = useRef<MediaStream | null>(null);
  const chunkTimerRef    = useRef<ReturnType<typeof setTimeout> | null>(null);
  const elapsedTimerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  // Accumule tous les chunks WebM pour une diarisation fullsize a la fin
  const allChunksRef     = useRef<Blob[]>([]);

  // Auth check
  useEffect(() => {
    getMe().then(setUser).catch(() => router.push("/login"));
  }, [router]);

  // ── Recording lifecycle ────────────────────────────────────────────────
  //
  // Pattern "stop + restart" pour chunker : chaque MediaRecorder produit UN
  // fichier WebM COMPLET (avec header) à chaque stop(). Utiliser timeslice sur
  // start() envoie des fragments sans header → Whisper ne peut décoder que le
  // 1er chunk, les suivants sont invalides silencieusement.
  const stoppingRef      = useRef(false);          // évite de redémarrer sur stop volontaire
  const buildRecorder = useCallback((stream: MediaStream): MediaRecorder => {
    const mr = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });

    // À chaque dataavailable → envoie chunk au STT
    mr.ondataavailable = async (e) => {
      if (e.data.size < 1000) return;              // ignore les chunks vides
      // Si diarize activee, on garde tous les blobs pour les ressortir en fin
      allChunksRef.current.push(e.data);
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
      }
    };

    // À chaque stop → si on n'est pas en train d'arrêter volontairement,
    // relancer un nouveau MediaRecorder pour le chunk suivant
    mr.onstop = () => {
      if (stoppingRef.current) return;             // stop volontaire (bouton Arrêter)
      if (!streamRef.current) return;
      const nextMr = buildRecorder(streamRef.current);
      mediaRecorderRef.current = nextMr;
      nextMr.start();
      // Reprogrammer le prochain stop dans 30s
      chunkTimerRef.current = setTimeout(() => nextMr.stop(), CHUNK_DURATION_MS);
    };

    return mr;
  }, []);

  const startRecording = useCallback(async () => {
    setError(null);
    setSegments([]);
    setSummary(null);
    setDiarizedResult(null);
    setSpeakerNames({});
    setElapsedSec(0);
    allChunksRef.current = [];
    stoppingRef.current = false;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mr = buildRecorder(stream);
      mediaRecorderRef.current = mr;
      mr.start();                                  // pas de timeslice
      setStep("recording");

      // Timer chunks : stop() après 30s → onstop restart un nouveau MR
      chunkTimerRef.current = setTimeout(() => mr.stop(), CHUNK_DURATION_MS);

      // Timer d'affichage temps écoulé
      elapsedTimerRef.current = setInterval(() => setElapsedSec(s => s + 1), 1000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Microphone inaccessible.");
      setStep("error");
    }
  }, [buildRecorder]);

  const stopRecording = useCallback(() => {
    // Marquer l'arrêt volontaire → onstop ne relancera pas de nouveau MR
    stoppingRef.current = true;
    if (chunkTimerRef.current) { clearTimeout(chunkTimerRef.current); chunkTimerRef.current = null; }
    if (elapsedTimerRef.current) { clearInterval(elapsedTimerRef.current); elapsedTimerRef.current = null; }
    mediaRecorderRef.current?.stop();
    streamRef.current?.getTracks().forEach(t => t.stop());
    setStep("stopped");
  }, []);

  // Cleanup au unmount
  useEffect(() => {
    return () => {
      stoppingRef.current = true;
      mediaRecorderRef.current?.stop();
      streamRef.current?.getTracks().forEach(t => t.stop());
      if (chunkTimerRef.current)   clearTimeout(chunkTimerRef.current);
      if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
    };
  }, []);

  // ── Diarisation : lance AssemblyAI sur l'audio complet accumule ───────
  const runDiarization = useCallback(async (): Promise<DiarizedTranscript | null> => {
    if (allChunksRef.current.length === 0) return null;
    setDiarizing(true);
    try {
      // Fusionne tous les blobs WebM en un seul fichier (chaque chunk est deja
      // un WebM complet avec son header grace au pattern stop+restart)
      const fullBlob = new Blob(allChunksRef.current, { type: "audio/webm" });
      const diarized = await transcribeDiarized(fullBlob, "fr");
      setDiarizedResult(diarized);
      // Init les noms des speakers avec "Intervenant X" par defaut
      const initNames: Record<string, string> = {};
      diarized.speakers.forEach(s => { initNames[s] = `Intervenant ${s}`; });
      setSpeakerNames(initNames);
      return diarized;
    } catch (e) {
      console.error("Diarisation error:", e);
      setError(e instanceof Error ? `Diarisation echouee : ${e.message}` : "Erreur diarisation");
      return null;
    } finally {
      setDiarizing(false);
    }
  }, []);

  // ── Summary ───────────────────────────────────────────────────────────
  const generateSummary = useCallback(async () => {
    setStep("summarizing");
    setError(null);

    // Si diarize active : lance d'abord la diarisation complete pour avoir
    // les speakers, puis construit un transcript "[Alice] ... [Bob] ..."
    let transcript: string;
    if (diarize) {
      const diarized = diarizedResult || await runDiarization();
      if (!diarized || diarized.segments.length === 0) {
        setError("Aucun segment diarise.");
        setStep("stopped");
        return;
      }
      transcript = formatDiarizedTranscript(diarized.segments, speakerNames);
    } else {
      transcript = segments.map(s => s.text).join(" ");
    }

    if (!transcript.trim()) {
      setError("Aucun transcript à résumer.");
      setStep("stopped");
      return;
    }

    try {
      const res = await summarizeMeeting(transcript, summaryLang, summaryStyle);
      setSummary(res);
      setStep("summarized");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur résumé");
      setStep("error");
    }
  }, [segments, summaryLang, summaryStyle, diarize, diarizedResult, speakerNames, runDiarization]);

  const downloadMarkdown = useCallback(() => {
    if (!summary) return;
    const blob = new Blob([summary.summary], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `meeting_${Date.now()}_${summary.target_lang}.md`;
    a.click();
  }, [summary]);

  const downloadMeetingPdf = useCallback(async () => {
    if (!summary) return;
    const { downloadPDF, diarizedToSrt } = await import("@/lib/exports");
    const subtitle = `Compte-rendu · ${SUMMARY_LANGS.find(l => l.code === summary.target_lang)?.label} · ${new Date().toLocaleDateString("fr-FR")}`;
    await downloadPDF({
      title:       "Compte-rendu de réunion",
      subtitle,
      segments:    diarizedResult ? diarizedToSrt(diarizedResult.segments, speakerNames) : undefined,
      sourceLabel: diarizedResult ? undefined : "Transcript",
      sourceText:  diarizedResult ? undefined : segments.map(s => s.text).join(" "),
      targetLabel: "Compte-rendu",
      targetText:  summary.summary,
      footerNote:  "Généré par traduction-audio.fr",
    }, `meeting_${Date.now()}_${summary.target_lang}.pdf`);
  }, [summary, diarizedResult, speakerNames, segments]);

  const downloadMeetingSrt = useCallback(async () => {
    if (!diarizedResult || diarizedResult.segments.length === 0) return;
    const { downloadSRT, diarizedToSrt } = await import("@/lib/exports");
    downloadSRT(
      diarizedToSrt(diarizedResult.segments, speakerNames),
      `meeting_${Date.now()}.srt`,
    );
  }, [diarizedResult, speakerNames]);

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
    <>
    <AppHeader current="meeting" />
    <main style={{
      minHeight: "calc(100vh - 65px)", background: "var(--background)",
      padding: "32px 24px 80px",
      display: "flex", flexDirection: "column", alignItems: "center",
    }}>
      <div style={{ width: "100%", maxWidth: "720px" }}>

        {/* Title bar */}
        <div style={{ marginBottom: "24px" }}>
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

              {/* Toggle diarisation (a activer AVANT de demarrer) */}
              <div style={{ marginTop: "20px", display: "flex", justifyContent: "center" }}>
                <button
                  onClick={() => setDiarize(d => !d)}
                  title="Identifier automatiquement les differents intervenants dans la reunion (+1,5 credit/min)"
                  style={{
                    display: "inline-flex", alignItems: "center", gap: "8px",
                    padding: "6px 14px", borderRadius: "999px", fontSize: "12px",
                    cursor: "pointer", transition: "all 0.2s",
                    background: diarize ? "rgba(201,169,110,0.12)" : "var(--surface)",
                    border: `1px solid ${diarize ? "var(--accent)" : "var(--border)"}`,
                    color: diarize ? "var(--accent)" : "var(--muted)",
                  }}
                >
                  <span style={{
                    width: "14px", height: "14px", borderRadius: "4px",
                    display: "inline-flex", alignItems: "center", justifyContent: "center",
                    background: diarize ? "var(--accent)" : "transparent",
                    border: `1px solid ${diarize ? "var(--accent)" : "var(--border)"}`,
                    color: "#0c0c0e", fontSize: "11px", lineHeight: 1,
                  }}>{diarize ? "✓" : ""}</span>
                  Identifier les intervenants
                  <span style={{ fontSize: "10px", opacity: 0.6 }}>+1,5 crédit/min</span>
                </button>
              </div>
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
              {/* Si diarisation faite : affichage par intervenant avec couleur */}
              {diarizedResult ? (
                diarizedResult.segments.map((seg, i) => {
                  const name = speakerNames[seg.speaker] || `Intervenant ${seg.speaker}`;
                  const color = speakerColor(seg.speaker);
                  return (
                    <div key={i} style={{
                      display: "grid", gridTemplateColumns: "auto 1fr", gap: "10px",
                      fontSize: "13px", lineHeight: 1.6, color: "var(--foreground)",
                    }}>
                      <div style={{ display: "flex", flexDirection: "column", gap: "2px", minWidth: "100px" }}>
                        <span style={{
                          padding: "1px 8px", borderRadius: "999px", fontSize: "10px", fontWeight: 500,
                          background: `${color}22`, border: `1px solid ${color}66`, color,
                          fontFamily: "ui-monospace, monospace", width: "fit-content",
                        }}>{name}</span>
                        <span style={{ fontSize: "9px", color: "var(--muted)", opacity: 0.7, fontFamily: "monospace" }}>
                          {fmtTs(seg.start)}
                        </span>
                      </div>
                      <div>{seg.text}</div>
                    </div>
                  );
                })
              ) : (
                segments.map((s, i) => (
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
                ))
              )}
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

            {/* Si diarisation activee : bouton pour la lancer manuellement AVANT le resume */}
            {diarize && !diarizedResult && !diarizing && segments.length > 0 && (
              <button onClick={() => runDiarization()} style={{
                width: "100%", padding: "10px", borderRadius: "10px", cursor: "pointer",
                fontSize: "13px", fontWeight: 500, marginBottom: "10px",
                background: "var(--surface)", border: "1px solid var(--accent-dim)",
                color: "var(--accent)",
              }}>
                🎙 Identifier les intervenants (envoie l&apos;audio complet)
              </button>
            )}

            {diarizing && (
              <p style={{ textAlign: "center", fontSize: "12px", color: "var(--muted)", marginBottom: "10px" }}>
                Identification des intervenants en cours…
              </p>
            )}

            {/* Mapping speakers → noms (apres diarisation) */}
            {diarizedResult && diarizedResult.speakers.length > 0 && (
              <div style={{
                marginBottom: "12px", padding: "14px",
                background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "10px",
              }}>
                <p style={{
                  fontSize: "11px", letterSpacing: "0.15em", textTransform: "uppercase",
                  color: "var(--muted)", margin: "0 0 10px 0",
                }}>
                  {diarizedResult.speakers.length} intervenant{diarizedResult.speakers.length > 1 ? "s" : ""} détecté{diarizedResult.speakers.length > 1 ? "s" : ""} — personnalisez les noms
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "8px" }}>
                  {diarizedResult.speakers.map(sp => (
                    <div key={sp} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <span style={{
                        padding: "2px 8px", borderRadius: "999px", fontSize: "10px", fontWeight: 500,
                        background: `${speakerColor(sp)}22`,
                        border: `1px solid ${speakerColor(sp)}66`,
                        color: speakerColor(sp),
                        fontFamily: "ui-monospace, monospace", flexShrink: 0,
                      }}>{sp}</span>
                      <input
                        type="text"
                        value={speakerNames[sp] || ""}
                        onChange={e => setSpeakerNames(prev => ({ ...prev, [sp]: e.target.value }))}
                        placeholder={`Intervenant ${sp}`}
                        style={{
                          flex: 1, minWidth: 0, padding: "6px 10px",
                          background: "var(--background)", border: "1px solid var(--border)",
                          borderRadius: "8px", fontSize: "13px", color: "var(--foreground)",
                          outline: "none",
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}

            <button onClick={generateSummary} disabled={step === "summarizing" || diarizing} style={{
              width: "100%", padding: "12px", borderRadius: "10px", cursor: (step === "summarizing" || diarizing) ? "wait" : "pointer",
              fontSize: "13px", fontWeight: 600,
              background: (step === "summarizing" || diarizing)
                ? "var(--border)"
                : "linear-gradient(135deg, var(--accent), var(--accent-dim))",
              color: (step === "summarizing" || diarizing) ? "var(--muted)" : "#0c0c0e",
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
              <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                <button onClick={downloadMarkdown} style={{
                  padding: "5px 12px", borderRadius: "8px", fontSize: "11px", cursor: "pointer",
                  background: "rgba(201,169,110,0.1)", border: "1px solid var(--accent-dim)", color: "var(--accent)",
                }}>
                  .md
                </button>
                <button onClick={downloadMeetingPdf} style={{
                  padding: "5px 12px", borderRadius: "8px", fontSize: "11px", cursor: "pointer",
                  background: "rgba(201,169,110,0.1)", border: "1px solid var(--accent-dim)", color: "var(--accent)",
                }}>
                  .pdf
                </button>
                {diarizedResult && diarizedResult.segments.length > 0 && (
                  <button onClick={downloadMeetingSrt} style={{
                    padding: "5px 12px", borderRadius: "8px", fontSize: "11px", cursor: "pointer",
                    background: "rgba(201,169,110,0.1)", border: "1px solid var(--accent-dim)", color: "var(--accent)",
                  }}>
                    .srt
                  </button>
                )}
              </div>
            </div>
            <pre style={{
              whiteSpace: "pre-wrap", fontFamily: "var(--font-body), ui-monospace, monospace",
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
    <FooterMinimal />
    </>
  );
}
