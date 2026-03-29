"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { transcribe, translate, synthesize } from "@/lib/api";
import type { TranscribeResult, TranslateResult } from "@/lib/api";

type Step = "idle" | "recording" | "transcribing" | "translating" | "done" | "error";

// Retourne un pourcentage basé sur le temps réel écoulé vs durée attendue
// Approche 95% asymptotiquement, saute à 100 quand done=true
function useProgress(active: boolean, done: boolean, expectedMs: number) {
  const [pct, setPct] = useState(0);
  const startRef = useRef<number | null>(null);
  const rafRef   = useRef<number | null>(null);

  useEffect(() => {
    if (active && !done) {
      startRef.current = performance.now();
      const tick = () => {
        const elapsed = performance.now() - (startRef.current ?? 0);
        // Courbe logarithmique : monte vite au début, ralentit vers 95%
        const raw = elapsed / expectedMs;
        const capped = 1 - Math.exp(-raw * 2.5); // asymptote à 1
        setPct(Math.min(Math.round(capped * 95), 95));
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } else if (done) {
      setPct(100);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    } else {
      setPct(0);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    }
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [active, done, expectedMs]);

  return pct;
}

function ProgressBar({ pct, label }: { pct: number; label: string }) {
  return (
    <div className="w-full space-y-3">
      <div className="flex justify-between items-center">
        <span className="text-sm" style={{ color: "var(--muted)" }}>{label}</span>
        <span className="text-sm font-medium tabular-nums" style={{ color: "var(--accent)" }}>{pct}%</span>
      </div>
      <div className="h-1 rounded-full overflow-hidden" style={{ background: "var(--border)" }}>
        <div className="h-full rounded-full transition-all duration-300"
          style={{
            width: `${pct}%`,
            background: "linear-gradient(90deg, var(--accent-dim), var(--accent))",
          }} />
      </div>
    </div>
  );
}

function ProcessingSteps({ step, sttDone }: { step: Step; sttDone: boolean }) {
  // STT : ~45s pour Whisper small sur un bulletin typique
  // LLM : ~2s pour Groq
  const sttPct = useProgress(step === "transcribing", sttDone, 45_000);
  const llmPct = useProgress(step === "translating", step === "done", 2_500);

  return (
    <div className="w-full py-8 space-y-6 animate-fade-up">
      <ProgressBar
        pct={sttDone ? 100 : sttPct}
        label="Transcription — Whisper"
      />
      <ProgressBar
        pct={step === "translating" ? llmPct : step === "done" ? 100 : 0}
        label="Translation — Llama"
      />
    </div>
  );
}

function StepIndicator({ step }: { step: Step }) {
  const steps = [
    { key: "transcribing", label: "Transcription" },
    { key: "translating",  label: "Translation" },
    { key: "done",         label: "Done" },
  ];
  const active = steps.findIndex(s =>
    step === "transcribing" ? s.key === "transcribing" :
    step === "translating"  ? s.key === "translating" :
    step === "done"         ? s.key === "done" : false
  );

  if (!["transcribing", "translating", "done"].includes(step)) return null;

  return (
    <div className="flex items-center gap-3 mb-10 animate-fade-up">
      {steps.map((s, i) => (
        <div key={s.key} className="flex items-center gap-3">
          <div className="flex flex-col items-center gap-1.5">
            <div className="w-2 h-2 rounded-full transition-all duration-500"
              style={{ background: i <= active ? "var(--accent)" : "var(--border)" }} />
            <span className="text-xs tracking-wide"
              style={{ color: i <= active ? "var(--accent)" : "var(--muted)" }}>
              {s.label}
            </span>
          </div>
          {i < steps.length - 1 && (
            <div className="w-12 h-px mb-4 transition-all duration-500"
              style={{ background: i < active ? "var(--accent)" : "var(--border)" }} />
          )}
        </div>
      ))}
    </div>
  );
}

export default function Home() {
  const [step, setStep]             = useState<Step>("idle");
  const [stt, setStt]               = useState<TranscribeResult | null>(null);
  const [llm, setLlm]               = useState<TranslateResult | null>(null);
  const [audioUrl, setAudioUrl]     = useState<string | null>(null);
  const [synthLoading, setSynthLoading] = useState(false);
  const [error, setError]           = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

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

  const stopRecording = () => mediaRef.current?.stop();

  const onSynthesize = async () => {
    if (!llm) return;
    setSynthLoading(true);
    try {
      const blob = await synthesize(llm.translation);
      setAudioUrl(URL.createObjectURL(blob));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSynthLoading(false);
    }
  };

  const reset = () => {
    setStep("idle");
    setStt(null);
    setLlm(null);
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

      {/* Step indicator (shown during processing + done) */}
      <StepIndicator step={step} />

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
        {(step === "transcribing" || step === "translating") && (
          <ProcessingSteps step={step} sttDone={!!stt} />
        )}

        {/* Results */}
        {step === "done" && stt && llm && (
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
                    {stt.language.toUpperCase()} · {Math.round(stt.language_probability * 100)}%
                  </span>
                </div>
                <p className="px-6 py-6 text-sm leading-7"
                  style={{ color: "var(--foreground)", fontWeight: 300 }}>{stt.text}</p>
              </div>

              {/* Translation card */}
              <div className="rounded-2xl" style={{ background: "var(--surface)", border: "1px solid var(--accent-dim)" }}>
                <div className="flex items-center justify-between px-6 pt-6 pb-4"
                  style={{ borderBottom: "1px solid rgba(201,169,110,0.2)" }}>
                  <span className="text-xs tracking-[0.2em] uppercase font-medium"
                    style={{ color: "var(--accent)" }}>English translation</span>
                  <span className="text-xs px-3 py-1 rounded-full"
                    style={{ background: "rgba(201,169,110,0.1)", color: "var(--accent)" }}>
                    {llm.latency_ms}ms · {llm.model.replace("groq/", "")}
                  </span>
                </div>
                <p className="px-6 py-6 text-sm leading-7"
                  style={{ color: "var(--foreground)", fontWeight: 300 }}>{llm.translation}</p>
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-col sm:flex-row gap-4 mt-2 mb-8">
              {!audioUrl ? (
                <button onClick={onSynthesize} disabled={synthLoading}
                  className="flex-1 py-3.5 rounded-xl text-sm font-medium tracking-wide transition-all hover:opacity-90 disabled:opacity-50 flex items-center justify-center gap-2"
                  style={{ background: "linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%)", color: "#0c0c0e" }}>
                  {synthLoading ? (
                    <>
                      <div className="w-4 h-4 rounded-full border-2 border-transparent"
                        style={{ borderTopColor: "#0c0c0e", animation: "spin 0.8s linear infinite" }} />
                      Generating audio…
                    </>
                  ) : "🔊 Listen in English"}
                </button>
              ) : (
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
              {stt.model} · {llm.prompt_version} · {Math.round(stt.duration)}s audio
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
          traduction-audio.fr · Whisper · Llama · MMS-TTS
        </p>
      </footer>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </main>
  );
}
