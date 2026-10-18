/**
 * Export des resultats : SRT (sous-titres), PDF (document imprimable), TXT.
 *
 * Toutes les fonctions sont cote client (navigator.download), aucune
 * dependance serveur. Idempotent, aucune donnee envoyee a un tiers.
 */
import type { TranslationSegment } from "@/lib/api";
import type { DiarizedSegment } from "@/lib/meeting";

// ── Helpers communs ─────────────────────────────────────────────────────────
function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Convertit un timestamp (secondes) en format SRT "HH:MM:SS,mmm". */
function fmtSrtTime(sec: number): string {
  const h  = Math.floor(sec / 3600);
  const m  = Math.floor((sec % 3600) / 60);
  const s  = Math.floor(sec % 60);
  const ms = Math.round((sec - Math.floor(sec)) * 1000);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")},${String(ms).padStart(3, "0")}`;
}

// ── SRT (sous-titres standard, compatible VLC, YouTube, Premiere…) ──────────

export interface SrtSegment {
  start: number;
  end:   number;
  text:  string;
  speaker?: string;   // optionnel — prefixe "[Nom] " ajoute si present
}

/**
 * Genere un fichier .srt a partir de segments horodates.
 * Format SRT standard : index, timecode, texte, ligne vide.
 */
export function downloadSRT(segments: SrtSegment[], filename: string): void {
  const srt = segments
    .map((seg, i) => {
      const prefix = seg.speaker ? `[${seg.speaker}] ` : "";
      return `${i + 1}\n${fmtSrtTime(seg.start)} --> ${fmtSrtTime(seg.end)}\n${prefix}${seg.text}\n`;
    })
    .join("\n");
  // BOM UTF-8 pour compatibilite Windows / certains lecteurs
  triggerDownload(new Blob(["﻿" + srt], { type: "application/x-subrip;charset=utf-8" }), filename);
}

/** Convertit des TranslationSegment (pipeline) en SrtSegment. */
export function segmentsToSrt(
  segments: TranslationSegment[],
  options: { useTranslation?: boolean; speakerNames?: Record<string, string> } = {},
): SrtSegment[] {
  const { useTranslation = true, speakerNames = {} } = options;
  return segments.map(s => ({
    start:   s.start,
    end:     s.end,
    text:    useTranslation ? s.translation : s.original,
    speaker: s.speaker ? (speakerNames[s.speaker] || `Intervenant ${s.speaker}`) : undefined,
  }));
}

/** Convertit des DiarizedSegment (/meeting) en SrtSegment. */
export function diarizedToSrt(
  segments: DiarizedSegment[],
  speakerNames: Record<string, string> = {},
): SrtSegment[] {
  return segments.map(s => ({
    start:   s.start,
    end:     s.end,
    text:    s.text,
    speaker: speakerNames[s.speaker] || `Intervenant ${s.speaker}`,
  }));
}

// ── PDF (jsPDF, cote client) ────────────────────────────────────────────────

export interface PdfExportOptions {
  title:         string;
  subtitle?:     string;         // ex: "Francais -> English — 2026-10-18"
  sourceText?:   string;
  sourceLabel?:  string;         // ex: "Transcription originale"
  targetText?:   string;
  targetLabel?:  string;         // ex: "Traduction (English)"
  segments?:     SrtSegment[];   // si fourni, remplace sourceText/targetText par un affichage speaker
  footerNote?:   string;         // ex: "Genere par traduction-audio.fr"
}

/**
 * Genere un PDF a partir d'un resultat de traduction ou d'un compte-rendu.
 * Mise en page sobre : header, meta, sections, pied de page.
 * jsPDF charge dynamiquement pour alleger le bundle initial.
 */
// Les polices standard jsPDF ne couvrent que le Latin-1 : sans police Unicode,
// une traduction en ukrainien (cyrillique) sortirait illisible.
const FONT = "DejaVu";
let fontCache: Promise<{ normal: string; bold: string }> | null = null;

async function fetchFontBase64(url: string): Promise<string> {
  const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
  let bin = "";
  for (let i = 0; i < buf.length; i += 0x8000) {
    bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

function loadFonts() {
  if (!fontCache) {
    fontCache = Promise.all([
      fetchFontBase64("/fonts/DejaVuSans.ttf"),
      fetchFontBase64("/fonts/DejaVuSans-Bold.ttf"),
    ]).then(([normal, bold]) => ({ normal, bold }));
    fontCache.catch(() => { fontCache = null; });
  }
  return fontCache;
}

export async function downloadPDF(opts: PdfExportOptions, filename: string): Promise<void> {
  const [{ jsPDF }, fonts] = await Promise.all([import("jspdf"), loadFonts()]);

  const doc = new jsPDF({ unit: "pt", format: "a4" });
  doc.addFileToVFS("DejaVuSans.ttf", fonts.normal);
  doc.addFileToVFS("DejaVuSans-Bold.ttf", fonts.bold);
  doc.addFont("DejaVuSans.ttf", FONT, "normal");
  doc.addFont("DejaVuSans-Bold.ttf", FONT, "bold");
  const pageW = doc.internal.pageSize.getWidth();
  const pageH = doc.internal.pageSize.getHeight();
  const margin = 48;
  const maxW = pageW - margin * 2;

  let y = margin;

  // ── Header ──
  doc.setFont(FONT, "bold");
  doc.setFontSize(18);
  doc.setTextColor(26, 22, 18);
  doc.text(opts.title, margin, y);
  y += 20;

  if (opts.subtitle) {
    doc.setFont(FONT, "normal");
    doc.setFontSize(10);
    doc.setTextColor(130, 125, 115);
    doc.text(opts.subtitle, margin, y);
    y += 14;
  }

  // ── Accent line ──
  doc.setDrawColor(201, 169, 110);
  doc.setLineWidth(1.2);
  doc.line(margin, y + 4, margin + 60, y + 4);
  y += 24;

  // ── Helper pour section texte avec wrapping et saut de page ──
  const addSection = (label: string, body: string, color: [number, number, number]) => {
    doc.setFont(FONT, "bold");
    doc.setFontSize(9);
    doc.setTextColor(...color);
    doc.text(label.toUpperCase(), margin, y);
    y += 14;

    doc.setFont(FONT, "normal");
    doc.setFontSize(11);
    doc.setTextColor(40, 36, 30);
    const lines = doc.splitTextToSize(body, maxW) as string[];
    for (const line of lines) {
      if (y > pageH - margin - 20) {
        doc.addPage();
        y = margin;
      }
      doc.text(line, margin, y);
      y += 15;
    }
    y += 10;
  };

  // ── Segments speaker (prioritaires si fournis) ──
  if (opts.segments && opts.segments.length > 0) {
    doc.setFont(FONT, "bold");
    doc.setFontSize(9);
    doc.setTextColor(201, 169, 110);
    doc.text("TRANSCRIPTION", margin, y);
    y += 16;

    for (const seg of opts.segments) {
      if (y > pageH - margin - 30) {
        doc.addPage();
        y = margin;
      }
      const speaker = seg.speaker ? `${seg.speaker} ` : "";
      const ts = `[${fmtSrtTime(seg.start).slice(0, 8)}]`;
      doc.setFont(FONT, "bold");
      doc.setFontSize(9);
      doc.setTextColor(201, 169, 110);
      doc.text(`${ts} ${speaker}`.trim(), margin, y);
      y += 12;
      doc.setFont(FONT, "normal");
      doc.setFontSize(11);
      doc.setTextColor(40, 36, 30);
      const lines = doc.splitTextToSize(seg.text, maxW) as string[];
      for (const line of lines) {
        if (y > pageH - margin - 20) { doc.addPage(); y = margin; }
        doc.text(line, margin, y);
        y += 14;
      }
      y += 6;
    }
  } else {
    if (opts.sourceText) {
      addSection(opts.sourceLabel || "Source", opts.sourceText, [130, 125, 115]);
    }
    if (opts.targetText) {
      addSection(opts.targetLabel || "Traduction", opts.targetText, [201, 169, 110]);
    }
  }

  // ── Footer sur chaque page ──
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    doc.setFont(FONT, "normal");
    doc.setFontSize(8);
    doc.setTextColor(160, 155, 145);
    const footer = opts.footerNote || "Généré par traduction-audio.fr";
    doc.text(footer, margin, pageH - 24);
    doc.text(`${i} / ${totalPages}`, pageW - margin, pageH - 24, { align: "right" });
  }

  doc.save(filename);
}
