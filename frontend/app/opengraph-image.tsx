import { ImageResponse } from "next/og";

export const alt = "traduction-audio.fr — Transcription et traduction audio par IA";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function OGImage() {
  return new ImageResponse(
    (
      <div
        style={{
          height: "100%", width: "100%",
          display: "flex", flexDirection: "column",
          alignItems: "flex-start", justifyContent: "center",
          padding: "80px",
          background: "linear-gradient(135deg, #0f0e0c 0%, #1a1612 100%)",
          color: "#f5efe4",
          fontFamily: "serif",
        }}
      >
        <div style={{ fontSize: 32, color: "#c9a96e", letterSpacing: 2, marginBottom: 24 }}>
          traduction-audio.fr
        </div>
        <div style={{ fontSize: 72, fontWeight: 400, lineHeight: 1.15, marginBottom: 24, maxWidth: 980 }}>
          Translate any voice, instantly.
        </div>
        <div style={{ fontSize: 28, color: "#a8a295", lineHeight: 1.4, maxWidth: 920 }}>
          Transcription + traduction + voix cloner — 30 langues. Diarisation, comptes-rendus de reunion.
        </div>
        <div style={{ display: "flex", gap: 16, marginTop: 48, fontSize: 20, color: "#c9a96e" }}>
          <span>Whisper IA</span>
          <span style={{ opacity: 0.4 }}>·</span>
          <span>GPT-4o</span>
          <span style={{ opacity: 0.4 }}>·</span>
          <span>AssemblyAI</span>
          <span style={{ opacity: 0.4 }}>·</span>
          <span>XTTS</span>
        </div>
      </div>
    ),
    { ...size },
  );
}
