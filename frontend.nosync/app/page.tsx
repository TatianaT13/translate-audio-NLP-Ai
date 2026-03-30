"use client";

import { useState, useRef, useCallback } from "react";
import { process as runPipeline, audioFromBase64 } from "@/lib/api";
import type { ProcessResult } from "@/lib/api";

type Step = "idle" | "recording" | "processing" | "done" | "error";

export default function Home() {
  const [step, setStep]     = useState<Step>("idle");
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError]   = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileRef   = useRef<HTMLInputElement>(null);
  const mediaRef  = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const run = useCallback(async (file: File | Blob) => {
    setError(null);
    setResult(null);
    setAudioUrl(null);
    setStep("processing");

    try {
      const res = await runPipeline(file);
      setResult(res);
      if (res.audio_b64) {
        const blob = audioFromBase64(res.audio_b64, res.audio_content_type);
        setAudioUrl(URL.createObjectURL(blob));
      }
      setStep("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStep("error");
    }
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
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
        stream.getTracks().forEach((t) => t.stop());
        run(blob);
      };
      mr.start();
      mediaRef.current = mr;
      setStep("recording");
    } catch {
      setError("Microphone inaccessible. Autorise l'accès dans ton navigateur.");
      setStep("error");
    }
  };

  const stopRecording = () => mediaRef.current?.stop();

  const reset = () => {
    setStep("idle");
    setResult(null);
    setAudioUrl(null);
    setError(null);
  };

  return (
    <main className="min-h-screen flex flex-col items-center px-4 py-20"
      style={{ background: "var(--background)" }}>

      {/* Header */}
      <header className="text-center mb-16 animate-fade-up max-w-2xl">
        <div className="inline-block text-xs tracking-[0.35em] uppercase mb-6 px-4 py-1.5 rounded-full"
          style={{ background: "rgba(201,169,110,0.08)", color: "var(--accent)" }}>
          AI Audio Translation
        </div>
        <h1 className="font-serif text-5xl md:text-6xl mb-6 leading-tight"
          style={{ color: "var(--foreground)" }}>
          Translate your voice.
          <br />
          <em style={{ color: "var(--accent)" }}>Instantly.</em>
        </h1>
        <p className="text-base leading-relaxed"
          style={{ color: "var(--muted)", fontWeight: 300, maxWidth: "38ch", margin: "0 auto" }}>
          Upload an audio file or record directly — we transcribe, translate, and read it back to you.
        </p>
      </header>

      <div className="w-full max-w-2xl">

        {/* Idle — upload + record */}
        {step === "idle" && (
          <div className="animate-fade-up space-y-0">
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              onClick={() => fileRef.current?.click()}
              className="rounded-2xl text-center cursor-pointer transition-all duration-300"
              style={{
                padding: "3.5rem 2rem",
                background: isDragging ? "rgba(201,169,110,0.06)" : "var(--surface)",
                border: `1.5px dashed ${isDragging ? "var(--accent)" : "var(--border)"}`,
              }}
            >
              <div className="text-4xl mb-5">🎵</div>
              <p className="text-base mb-2" style={{ color: "var(--foreground)", fontWeight: 400 }}>
                Drop an audio file here
              </p>
              <p className="text-sm" style={{ color: "var(--muted)" }}>
                MP3, WAV, M4A, OGG — any format
              </p>
              <input ref={fileRef} type="file" accept="audio/*" className="hidden"
                onChange={(e) => e.target.files?.[0] && run(e.target.files[0])} />
            </div>

            <div className="flex items-center gap-4 py-10">
              <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
              <span className="text-xs tracking-widest px-2" style={{ color: "var(--muted)" }}>OR</span>
              <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
            </div>

            <button
              onClick={startRecording}
              className="w-full py-4 rounded-2xl text-sm font-medium tracking-wide transition-all duration-300 hover:opacity-90 active:scale-[0.99]"
              style={{
                background: "linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%)",
                color: "#0c0c0e",
              }}
            >
              🎙 Record with microphone
            </button>
          </div>
        )}

        {/* Recording */}
        {step === "recording" && (
          <div className="text-center py-8 animate-fade-up">
            <div className="flex items-end justify-center gap-2 h-16 mb-8">
              {[0.0, 0.1, 0.2, 0.3, 0.15, 0.25, 0.2].map((delay, i) => (
                <div key={i} className="wave-bar w-2 rounded-full"
                  style={{ height: "100%", background: "var(--accent)", animationDelay: `${delay}s` }} />
              ))}
            </div>
            <p className="text-sm mb-2" style={{ color: "var(--foreground)" }}>Recording…</p>
            <p className="text-xs mb-8" style={{ color: "var(--muted)" }}>Speak clearly into your microphone</p>
            <button onClick={stopRecording}
              className="px-8 py-3 rounded-full text-sm font-medium transition-all hover:opacity-80"
              style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--foreground)" }}>
              ⏹ Stop & Translate
            </button>
          </div>
        )}

        {/* Processing */}
        {step === "processing" && (
          <div className="text-center py-16 animate-fade-up">
            <div className="flex items-center justify-center gap-3 mb-6">
              {["STT", "LLM", "TTS"].map((label, i) => (
                <div key={label} className="flex items-center gap-3">
                  <div className="flex flex-col items-center gap-2">
                    <div className="w-2 h-2 rounded-full"
                      style={{ background: "var(--accent)", animation: `pulse 1.4s ease-in-out ${i * 0.3}s infinite` }} />
                    <span className="text-xs tracking-wide" style={{ color: "var(--muted)" }}>{label}</span>
                  </div>
                  {i < 2 && <div className="w-8 h-px mb-4" style={{ background: "var(--border)" }} />}
                </div>
              ))}
            </div>
            <p className="text-sm" style={{ color: "var(--muted)" }}>Pipeline running…</p>
          </div>
        )}

        {/* Results */}
        {step === "done" && result && (
          <div className="animate-fade-up">

            {/* Cards côte à côte */}
            <div className="grid md:grid-cols-2 gap-6 mb-8">

              {/* Transcription card */}
              <div className="rounded-2xl" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                <div className="flex items-center justify-between px-6 pt-6 pb-4"
                  style={{ borderBottom: "1px solid var(--border)" }}>
                  <span className="text-xs tracking-[0.2em] uppercase font-medium"
                    style={{ color: "var(--muted)" }}>Original transcript</span>
                  <span className="text-xs px-3 py-1 rounded-full"
                    style={{ background: "rgba(201,169,110,0.1)", color: "var(--accent)" }}>
                    {result.language.toUpperCase()} · {Math.round(result.language_prob * 100)}%
                  </span>
                </div>
                <p className="px-6 py-6 text-sm leading-7"
                  style={{ color: "var(--foreground)", fontWeight: 300 }}>{result.source_text}</p>
              </div>

              {/* Translation card */}
              <div className="rounded-2xl" style={{ background: "var(--surface)", border: "1px solid var(--accent-dim)" }}>
                <div className="flex items-center justify-between px-6 pt-6 pb-4"
                  style={{ borderBottom: "1px solid rgba(201,169,110,0.2)" }}>
                  <span className="text-xs tracking-[0.2em] uppercase font-medium"
                    style={{ color: "var(--accent)" }}>English translation</span>
                  <span className="text-xs px-3 py-1 rounded-full"
                    style={{ background: "rgba(201,169,110,0.1)", color: "var(--accent)" }}>
                    {result.latency_llm_ms}ms
                  </span>
                </div>
                <p className="px-6 py-6 text-sm leading-7"
                  style={{ color: "var(--foreground)", fontWeight: 300 }}>{result.translation}</p>
              </div>
            </div>

            {/* Audio player + actions */}
            <div className="flex flex-col sm:flex-row gap-4 mt-2 mb-8">
              {audioUrl && (
                <div className="flex-1 rounded-xl px-4 py-2 flex items-center gap-3"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                  <span className="text-xs" style={{ color: "var(--muted)" }}>🔊</span>
                  <audio controls src={audioUrl} className="flex-1 h-8" style={{ accentColor: "var(--accent)" }} />
                </div>
              )}
              <button onClick={reset}
                className="px-6 py-3.5 rounded-xl text-sm transition-all hover:opacity-80"
                style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--muted)" }}>
                ↩ New translation
              </button>
            </div>

            {/* Meta */}
            <p className="text-center text-xs" style={{ color: "var(--muted)", opacity: 0.6 }}>
              STT {result.latency_stt_ms}ms · LLM {result.latency_llm_ms}ms · TTS {result.latency_tts_ms}ms · total {result.latency_total_ms}ms
            </p>
          </div>
        )}

        {/* Error */}
        {step === "error" && error && (
          <div className="text-center py-8 animate-fade-up">
            <div className="w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-5"
              style={{ background: "rgba(232,112,112,0.1)", border: "1px solid rgba(232,112,112,0.2)" }}>
              <span style={{ color: "#e87070" }}>✕</span>
            </div>
            <p className="text-sm mb-2" style={{ color: "#e87070" }}>Something went wrong</p>
            <p className="text-xs mb-8 max-w-sm mx-auto" style={{ color: "var(--muted)" }}>{error}</p>
            <button onClick={reset}
              className="px-6 py-3 rounded-full text-sm transition-all hover:opacity-80"
              style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--foreground)" }}>
              ↩ Try again
            </button>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="mt-20 text-center">
        <p className="text-xs tracking-wider" style={{ color: "var(--muted)", opacity: 0.4 }}>
          traduction-audio.fr · Whisper · Llama · Voxtral
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
