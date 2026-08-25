"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { getMe, logout } from "@/lib/auth";
import type { User } from "@/lib/auth";

const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8004";

type Status = "idle" | "connecting" | "listening" | "speaking" | "error";

const ALL_LANGS = [
  { code: "fr", label: "Français",  flag: "🇫🇷" },
  { code: "en", label: "Anglais",   flag: "🇬🇧" },
  { code: "es", label: "Espagnol",  flag: "🇪🇸" },
  { code: "de", label: "Allemand",  flag: "🇩🇪" },
  { code: "it", label: "Italien",   flag: "🇮🇹" },
  { code: "uk", label: "Ukrainien", flag: "🇺🇦" },
];

interface Line { role: "user" | "assistant"; text: string; ts: string; }

const STATUS_LABEL: Record<Status, string> = {
  idle:       "Prêt",
  connecting: "Connexion…",
  listening:  "À l'écoute",
  speaking:   "Traduction en cours",
  error:      "Erreur",
};

const STATUS_COLOR: Record<Status, string> = {
  idle:       "var(--muted)",
  connecting: "var(--accent-dim)",
  listening:  "#7BA05B",
  speaking:   "var(--accent)",
  error:      "#C67B4A",
};

export default function LivePage() {
  const router = useRouter();

  const [user,       setUser]       = useState<User | null>(null);
  const [status,     setStatus]     = useState<Status>("idle");
  const [sourceLang, setSourceLang] = useState("fr");
  const [targetLang, setTargetLang] = useState("en");
  const [error,      setError]      = useState<string | null>(null);
  const [lines,      setLines]      = useState<Line[]>([]);
  const [elapsedSec, setElapsedSec] = useState(0);

  const pcRef       = useRef<RTCPeerConnection | null>(null);
  const dcRef       = useRef<RTCDataChannel | null>(null);
  const streamRef   = useRef<MediaStream | null>(null);
  const elapsedRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const audioRef    = useRef<HTMLAudioElement | null>(null);

  useEffect(() => { getMe().then(setUser).catch(() => router.push("/login")); }, [router]);

  // Empêche source === target : si le user change la source pour la même que la target,
  // on décale automatiquement la target vers la 1re langue différente.
  useEffect(() => {
    if (sourceLang === targetLang) {
      const other = ALL_LANGS.find(l => l.code !== sourceLang);
      if (other) setTargetLang(other.code);
    }
  }, [sourceLang, targetLang]);

  // Auto-scroll transcript to bottom on new line
  useEffect(() => {
    if (scrollerRef.current) scrollerRef.current.scrollTop = scrollerRef.current.scrollHeight;
  }, [lines]);

  const stop = useCallback(() => {
    if (elapsedRef.current) { clearInterval(elapsedRef.current); elapsedRef.current = null; }
    try { dcRef.current?.close(); } catch {}
    try { pcRef.current?.close(); } catch {}
    streamRef.current?.getTracks().forEach(t => t.stop());
    dcRef.current = null;
    pcRef.current = null;
    streamRef.current = null;
    setStatus("idle");
  }, []);


  const start = useCallback(async () => {
    setError(null);
    setLines([]);
    setElapsedSec(0);
    setStatus("connecting");

    try {
      // 1) Ephemeral token depuis notre backend (la vraie clé OpenAI reste serveur)
      const token = localStorage.getItem("access_token");
      const r = await fetch(`${GATEWAY_URL}/realtime/session`, {
        method:  "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `Session HTTP ${r.status}`);
      }
      const { client_secret, model, session_id } = await r.json();

      // 2) PeerConnection + micro + audio out
      const pc = new RTCPeerConnection();
      pcRef.current = pc;

      // L'élément <audio> DOIT être dans le DOM pour être autorisé à jouer.
      // On utilise un ref sur un <audio hidden autoPlay> déclaré dans le JSX.
      pc.ontrack = (e) => {
        if (audioRef.current) audioRef.current.srcObject = e.streams[0];
      };

      // Constraints agressives contre l'auto-écho (le model entend sa propre voix
      // via speaker → re-traduction en boucle). Casque = mieux, mais on tente de limiter.
      const ms = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl:  true,
          channelCount:     1,
        },
      });
      streamRef.current = ms;
      pc.addTrack(ms.getTracks()[0]);

      // 3) Data channel pour events (config, transcripts)
      const dc = pc.createDataChannel("oai-events");
      dcRef.current = dc;

      const langLabel    = ALL_LANGS.find(l => l.code === targetLang)?.label ?? targetLang.toUpperCase();
      const sourceLabel  = ALL_LANGS.find(l => l.code === sourceLang)?.label ?? sourceLang.toUpperCase();

      dc.addEventListener("open", () => {
        // Config session déjà envoyée via multipart à /calls.
        setStatus("listening");
        elapsedRef.current = setInterval(() => setElapsedSec(s => s + 1), 1000);
      });

      dc.addEventListener("message", (e) => {
        try {
          const evt = JSON.parse(e.data);
          if (typeof window !== "undefined") console.debug("[realtime]", evt.type, evt);
          const now = new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

          // Input (ce que dit l'utilisateur) — noms variants selon versions API
          if (evt.type === "conversation.item.input_audio_transcription.completed" ||
              evt.type === "conversation.item.input_audio_transcription.done") {
            setLines(prev => [...prev, { role: "user", text: evt.transcript ?? "", ts: now }]);
          }
          // Output (traduction) — événements delta accumulés OU done final
          else if (evt.type === "response.audio_transcript.done" ||
                   evt.type === "response.output_audio_transcript.done") {
            setLines(prev => [...prev, { role: "assistant", text: evt.transcript ?? evt.text ?? "", ts: now }]);
          }
          // Model commence à parler → mute mic pour éviter feedback loop
          else if (evt.type === "response.audio.delta" ||
                   evt.type === "response.output_audio.delta" ||
                   evt.type === "response.created") {
            streamRef.current?.getAudioTracks().forEach(t => { t.enabled = false; });
            setStatus("speaking");
          }
          // Model a fini → réactive le mic
          else if (evt.type === "response.done" ||
                   evt.type === "response.completed" ||
                   evt.type === "output_audio_buffer.stopped") {
            streamRef.current?.getAudioTracks().forEach(t => { t.enabled = true; });
            setStatus("listening");
          }
          else if (evt.type === "error" || evt.type === "session.error") {
            setError(evt.error?.message ?? "Erreur OpenAI");
          }
        } catch { /* ignore parse errors */ }
      });

      // 4) SDP offer → OpenAI → answer.
      // Note : endpoint /v1/realtime/calls depuis oct 2025 (l'ancien /v1/realtime?model= a été deprecated).
      // Le model est déjà attaché au client_secret côté session.
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Endpoint GA (oct 2025+) : /v1/realtime/calls en MULTIPART form-data.
      // 2 champs : sdp (raw string du offer) + session (JSON stringifié).
      // Le Content-Type multipart/form-data est ajouté automatiquement par FormData.
      const sessionConfig = {
        type:  "realtime",
        model,
        output_modalities: ["audio"],   // impératif : sinon le model peut rester muet
        audio: {
          input: {
            // Force la langue source : sans hint, Whisper hallucine sur les accents.
            transcription: { model: "whisper-1", language: sourceLang },
            // VAD auto mais avec seuil et silence plus stricts pour éviter les faux positifs
            // sur du bruit ambiant / respiration / clics.
            turn_detection: {
              type: "server_vad",
              threshold: 0.7,
              prefix_padding_ms:   300,
              silence_duration_ms: 700,
              create_response:     true,
              interrupt_response:  true,
            },
          },
          output: { voice: "alloy" },
        },
        instructions:
          `You are a live simultaneous interpreter. The user speaks ${sourceLabel} (${sourceLang.toUpperCase()}). ` +
          `Your ONLY job is to translate their ${sourceLabel} speech into ${langLabel} (${targetLang.toUpperCase()}).\n\n` +
          `ABSOLUTE RULES:\n` +
          `1. NEVER answer questions. NEVER hold a conversation. NEVER add commentary.\n` +
          `2. NEVER introduce yourself, greet, or explain what you are doing.\n` +
          `3. If the user asks you a question, translate the question — do not answer it.\n` +
          `4. Translate FAITHFULLY — do not add or invent content.\n` +
          `5. If the input is unclear, noisy, silent, or nonsensical (random syllables, single words with no meaning), stay COMPLETELY SILENT. Do not fill.\n` +
          `6. If the transcription looks corrupt or in a wrong language, stay silent.\n` +
          `7. Keep the same tone, pacing, and emotion as the speaker.\n\n` +
          `Output only the ${langLabel} translation, spoken naturally. Nothing else.`,
      };
      const fd = new FormData();
      fd.set("sdp", offer.sdp ?? "");
      fd.set("session", JSON.stringify(sessionConfig));

      const sdpResp = await fetch("https://api.openai.com/v1/realtime/calls", {
        method:  "POST",
        headers: { Authorization: `Bearer ${client_secret}` },
        body:    fd,
      });
      if (!sdpResp.ok) {
        const body = await sdpResp.text().catch(() => "");
        throw new Error(`SDP HTTP ${sdpResp.status} (model=${model}) — ${body.slice(0, 300)}`);
      }
      const answerSdp = await sdpResp.text();
      await pc.setRemoteDescription({ type: "answer" as const, sdp: answerSdp });
    } catch (e) {
      setError((e as Error).message);
      setStatus("error");
      stop();
    }
  }, [targetLang, sourceLang, stop]);

  useEffect(() => () => stop(), [stop]);

  const running = status !== "idle" && status !== "error";
  const mmss = `${String(Math.floor(elapsedSec / 60)).padStart(2, "0")}:${String(elapsedSec % 60).padStart(2, "0")}`;

  return (
    <div style={{ minHeight: "100vh", background: "var(--background)", color: "var(--foreground)", display: "flex", flexDirection: "column" }}>
      {/* ── Header ── */}
      <header style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "16px 32px", borderBottom: "1px solid var(--border)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <button onClick={() => router.push("/")} style={{
            background: "transparent", border: "1px solid var(--border)", color: "var(--muted)",
            padding: "6px 12px", borderRadius: "6px", cursor: "pointer", fontSize: "12px",
          }}>← Accueil</button>
          <h1 className="font-serif" style={{ fontSize: "18px", color: "var(--accent)", letterSpacing: "0.02em" }}>
            🎙 Live · Traduction simultanée
          </h1>
          <span style={{
            fontSize: "10px", padding: "2px 8px", borderRadius: "999px",
            background: "rgba(201,169,110,0.12)", color: "var(--accent)", letterSpacing: "0.15em",
          }}>BÊTA</span>
        </div>
        {user && (
          <div style={{ display: "flex", alignItems: "center", gap: "12px", fontSize: "12px" }}>
            <span style={{ color: "var(--muted)" }}>{user.email}</span>
            <button onClick={() => logout().then(() => router.push("/login"))} style={{
              background: "transparent", border: "1px solid var(--border)", color: "var(--muted)",
              padding: "6px 12px", borderRadius: "6px", cursor: "pointer", fontSize: "11px",
            }}>Déconnexion</button>
          </div>
        )}
      </header>

      {/* ── Body ── */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", padding: "40px 24px", gap: "32px" }}>

        {/* Note fonctionnement */}
        {!running && (
          <p style={{
            fontSize: "11px", color: "var(--muted)", textAlign: "center", maxWidth: "520px", letterSpacing: "0.03em",
          }}>
            Le micro se coupe automatiquement pendant la traduction pour éviter les boucles.
          </p>
        )}

        {/* Sélecteurs source → target */}
        <div style={{ display: "flex", flexDirection: "column", gap: "12px", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap", justifyContent: "center" }}>
            <span style={{ fontSize: "10px", color: "var(--muted)", letterSpacing: "0.15em", textTransform: "uppercase" }}>Je parle</span>
            <select
              value={sourceLang}
              onChange={(e) => setSourceLang(e.target.value)}
              disabled={running}
              style={{
                padding: "8px 14px", borderRadius: "999px", fontSize: "13px",
                background: "var(--surface)", color: "var(--foreground)",
                border: "1px solid var(--border)", cursor: running ? "not-allowed" : "pointer",
                opacity: running ? 0.5 : 1,
              }}>
              {ALL_LANGS.map(l => (
                <option key={l.code} value={l.code}>{l.flag} {l.label}</option>
              ))}
            </select>

            <span style={{ fontSize: "16px", color: "var(--accent)" }}>→</span>

            <span style={{ fontSize: "10px", color: "var(--muted)", letterSpacing: "0.15em", textTransform: "uppercase" }}>Traduire vers</span>
            <select
              value={targetLang}
              onChange={(e) => setTargetLang(e.target.value)}
              disabled={running}
              style={{
                padding: "8px 14px", borderRadius: "999px", fontSize: "13px",
                background: "var(--surface)", color: "var(--foreground)",
                border: "1px solid var(--accent-dim)", cursor: running ? "not-allowed" : "pointer",
                opacity: running ? 0.5 : 1,
              }}>
              {ALL_LANGS.filter(l => l.code !== sourceLang).map(l => (
                <option key={l.code} value={l.code}>{l.flag} {l.label}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Statut + Bouton central */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "24px" }}>
          <div style={{
            display: "flex", alignItems: "center", gap: "10px",
            fontSize: "12px", color: STATUS_COLOR[status], letterSpacing: "0.12em",
          }}>
            <span style={{
              width: "8px", height: "8px", borderRadius: "50%", background: STATUS_COLOR[status],
              animation: running ? "pulse-ring 1.5s ease-in-out infinite" : "none",
            }} />
            <span style={{ textTransform: "uppercase" }}>{STATUS_LABEL[status]}</span>
            {running && <span className="tabular" style={{ color: "var(--muted)", marginLeft: "8px" }}>{mmss}</span>}
          </div>

          {!running ? (
            <button onClick={start} style={{
              padding: "18px 42px", borderRadius: "999px", fontSize: "15px", cursor: "pointer",
              background: "var(--accent)", color: "var(--background)", border: "none", fontWeight: 500,
              letterSpacing: "0.05em", boxShadow: "0 4px 24px rgba(201,169,110,0.25)",
            }}>
              ▶ Démarrer la conversation
            </button>
          ) : (
            <button onClick={stop} style={{
              padding: "18px 42px", borderRadius: "999px", fontSize: "15px", cursor: "pointer",
              background: "transparent", color: "var(--accent)", border: "1px solid var(--accent-dim)",
              letterSpacing: "0.05em",
            }}>
              ⬛ Arrêter
            </button>
          )}

          {error && (
            <div style={{
              padding: "12px 20px", borderRadius: "8px", maxWidth: "500px", textAlign: "center",
              background: "rgba(198,123,74,0.1)", border: "1px solid rgba(198,123,74,0.4)", color: "#C67B4A", fontSize: "13px",
            }}>
              {error}
            </div>
          )}
        </div>

        {/* Transcript live */}
        <div ref={scrollerRef} style={{
          width: "100%", maxWidth: "780px", flex: 1, minHeight: "220px", maxHeight: "50vh",
          overflowY: "auto", padding: "20px", border: "1px solid var(--border)", borderRadius: "12px",
          background: "var(--surface)",
        }}>
          {lines.length === 0 ? (
            <p style={{ color: "var(--muted)", textAlign: "center", fontSize: "13px", padding: "40px 0" }}>
              {running
                ? "Parle — la traduction apparaîtra ici en temps réel."
                : "Le transcript s'affichera pendant la conversation."}
            </p>
          ) : lines.map((line, i) => (
            <div key={i} style={{
              marginBottom: "12px", paddingBottom: "12px",
              borderBottom: i < lines.length - 1 ? "1px solid var(--border)" : "none",
            }}>
              <div style={{
                fontSize: "10px", letterSpacing: "0.15em", marginBottom: "4px",
                color: line.role === "user" ? "var(--muted)" : "var(--accent)",
              }}>
                {line.role === "user" ? "▸ VOUS" : "◂ TRADUCTION"} · {line.ts}
              </div>
              <div style={{ fontSize: "14px", lineHeight: 1.5, color: "var(--foreground)" }}>
                {line.text}
              </div>
            </div>
          ))}
        </div>

      </main>

      {/* <audio> caché mais DANS le DOM — indispensable pour que le browser autorise la lecture WebRTC */}
      <audio ref={audioRef} autoPlay playsInline hidden />
    </div>
  );
}
