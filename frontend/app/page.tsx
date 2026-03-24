"use client";

import { useState, useRef, useCallback } from "react";
import { transcribe, translate, synthesize } from "@/lib/api";
import type { TranscribeResult, TranslateResult } from "@/lib/api";

type Step = "idle" | "recording" | "transcribing" | "translating" | "done" | "error";

export default function Home() {
  const [step, setStep]               = useState<Step>("idle");
  const [stt, setStt]                 = useState<TranscribeResult | null>(null);
  const [llm, setLlm]                 = useState<TranslateResult | null>(null);
  const [audioUrl, setAudioUrl]       = useState<string | null>(null);
  const [error, setError]             = useState<string | null>(null);
  const [isDragging, setIsDragging]   = useState(false);

  const fileRef   = useRef<HTMLInputElement>(null);
  const mediaRef  = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const run = useCallback(async (file: File | Blob) => {
    setError(null);
    setStt(null);
    setLlm(null);
    setAudioUrl(null);

    try {
      setStep("transcribing");
      const sttResult = await transcribe(file);
      setStt(sttResult);

      setStep("translating");
      const llmResult = await translate(sttResult.text);
      setLlm(llmResult);

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

  const stopRecording = () => {
    mediaRef.current?.stop();
  };

  const onSynthesize = async () => {
    if (!llm) return;
    try {
      const blob = await synthesize(llm.translation);
      setAudioUrl(URL.createObjectURL(blob));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const reset = () => { setStep("idle"); setStt(null); setLlm(null); setAudioUrl(null); setError(null); };
  const isProcessing = step === "transcribing" || step === "translating";

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4 py-16" style={{ background: "var(--background)" }}>

      {/* Header */}
      <header className="text-center mb-16 animate-fade-up">
        <div className="text-xs tracking-[0.3em] uppercase mb-5" style={{ color: "var(--accent-dim)" }}>
          AI Audio Translation
        </div>
        <h1 className="font-serif text-5xl md:text-7xl mb-5 leading-tight" style={{ color: "var(--foreground)" }}>
          Translate your voice.
          <br />
          <em style={{ color: "var(--accent)", fontStyle: "italic" }}>Instantly.</em>
        </h1>
        <p className="text-base md:text-lg" style={{ color: "var(--muted)", fontWeight: 300 }}>
          Upload an audio file or record directly — we handle the rest.
        </p>
      </header>

      {/* Idle — upload + record */}
      {step === "idle" && (
        <div className="w-full max-w-xl animate-fade-up">
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
            className="rounded-2xl p-12 text-center cursor-pointer transition-all duration-300"
            style={{
              background: isDragging ? "rgba(201,169,110,0.06)" : "var(--surface)",
              border: `1.5px dashed ${isDragging ? "var(--accent)" : "var(--border)"}`,
            }}
          >
            <div className="text-4xl mb-4">🎵</div>
            <p className="text-sm mb-1" style={{ color: "var(--foreground)" }}>Drop an audio file here</p>
            <p className="text-xs" style={{ color: "var(--muted)" }}>MP3, WAV, M4A, OGG — any format</p>
            <input ref={fileRef} type="file" accept="audio/*" className="hidden"
              onChange={(e) => e.target.files?.[0] && run(e.target.files[0])} />
          </div>

          <div className="flex items-center gap-4 my-6">
            <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
            <span className="text-xs tracking-widest" style={{ color: "var(--muted)" }}>OR</span>
            <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
          </div>

          <button
            onClick={startRecording}
            className="w-full py-4 rounded-2xl text-sm font-medium tracking-wide transition-all duration-300 hover:opacity-90"
            style={{ background: "linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%)", color: "#0c0c0e" }}
          >
            🎙 Record with microphone
          </button>
        </div>
      )}

      {/* Recording */}
      {step === "recording" && (
        <div className="text-center animate-fade-up">
          <div className="flex items-end justify-center gap-1.5 h-14 mb-8">
            {[0.1, 0.2, 0.3, 0.15, 0.25, 0.2, 0.1].map((delay, i) => (
              <div key={i} className="wave-bar w-1.5 rounded-full"
                style={{ height: "100%", background: "var(--accent)", animationDelay: `${delay}s` }} />
            ))}
          </div>
          <p className="text-sm mb-6" style={{ color: "var(--muted)" }}>Recording in progress…</p>
          <button onClick={stopRecording}
            className="px-8 py-3 rounded-full text-sm font-medium transition-all hover:opacity-80"
            style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--foreground)" }}>
            ⏹ Stop & Translate
          </button>
        </div>
      )}

      {/* Processing */}
      {isProcessing && (
        <div className="text-center animate-fade-up">
          <div className="relative w-14 h-14 mx-auto mb-8">
            <div className="absolute inset-0 rounded-full animate-pulse-ring" style={{ background: "var(--accent)" }} />
            <div className="absolute inset-2 rounded-full" style={{ background: "var(--accent)" }} />
          </div>
          <p className="text-sm" style={{ color: "var(--muted)" }}>
            {step === "transcribing" ? "Transcribing audio…" : "Translating with AI…"}
          </p>
        </div>
      )}

      {/* Results */}
      {step === "done" && stt && llm && (
        <div className="w-full max-w-3xl animate-fade-up">
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="rounded-2xl p-6" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="flex items-center gap-2 mb-4">
                <span className="text-xs tracking-widest uppercase" style={{ color: "var(--muted)" }}>Original</span>
                <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: "rgba(201,169,110,0.1)", color: "var(--accent)" }}>
                  {stt.language.toUpperCase()} · {Math.round(stt.language_probability * 100)}%
                </span>
              </div>
              <p className="text-sm leading-relaxed" style={{ color: "var(--foreground)", fontWeight: 300 }}>{stt.text}</p>
            </div>

            <div className="rounded-2xl p-6" style={{ background: "var(--surface)", border: "1px solid var(--accent-dim)" }}>
              <div className="flex items-center gap-2 mb-4">
                <span className="text-xs tracking-widest uppercase" style={{ color: "var(--muted)" }}>Translation</span>
                <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: "rgba(201,169,110,0.1)", color: "var(--accent)" }}>
                  EN · {llm.latency_ms}ms
                </span>
              </div>
              <p className="text-sm leading-relaxed" style={{ color: "var(--foreground)", fontWeight: 300 }}>{llm.translation}</p>
            </div>
          </div>

          <div className="flex gap-3 justify-center flex-wrap">
            {!audioUrl ? (
              <button onClick={onSynthesize}
                className="px-6 py-3 rounded-full text-sm font-medium transition-all hover:opacity-90"
                style={{ background: "linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%)", color: "#0c0c0e" }}>
                🔊 Listen in English
              </button>
            ) : (
              <audio controls src={audioUrl} className="h-10" />
            )}
            <button onClick={reset}
              className="px-6 py-3 rounded-full text-sm transition-all hover:opacity-80"
              style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--muted)" }}>
              ↩ Translate another
            </button>
          </div>

          <p className="text-center text-xs mt-6" style={{ color: "var(--muted)" }}>
            {stt.model} · {llm.model.replace("groq/", "")} · {llm.prompt_version} · {Math.round(stt.duration)}s
          </p>
        </div>
      )}

      {/* Error */}
      {step === "error" && error && (
        <div className="text-center animate-fade-up max-w-md">
          <p className="text-sm mb-4" style={{ color: "#e87070" }}>{error}</p>
          <button onClick={reset}
            className="px-6 py-3 rounded-full text-sm transition-all hover:opacity-80"
            style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--muted)" }}>
            ↩ Try again
          </button>
        </div>
      )}

      {/* Footer */}
      <footer className="mt-24 text-center">
        <p className="text-xs tracking-wider" style={{ color: "var(--muted)" }}>
          traduction-audio.fr · Whisper · Llama · MMS-TTS
        </p>
      </footer>
    </main>
  );
}
