"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AppHeader } from "@/components/AppHeader";
import { FooterMinimal } from "@/components/Footer";
import {
  listHistory, getHistoryItem, deleteHistoryItem, clearHistory, historyAsText, isMeeting,
  type HistoryMeta, type HistoryItem,
} from "@/lib/history";

const KIND_LABEL: Record<string, string> = { translation: "Traduction", meeting: "Compte-rendu" };

const fmtDate = (iso: string | null) =>
  iso ? new Date(iso).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" }) : "";

const fmtDuration = (s: number | null) =>
  s == null ? "" : s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;

export default function HistoriquePage() {
  const [items, setItems]         = useState<HistoryMeta[] | null>(null);
  const [retention, setRetention] = useState(90);
  const [error, setError]         = useState<string | null>(null);
  const [openId, setOpenId]       = useState<string | null>(null);
  const [details, setDetails]     = useState<Record<string, HistoryItem | { error: string }>>({});
  const [copiedId, setCopiedId]   = useState<string | null>(null);

  useEffect(() => {
    listHistory()
      .then((r) => { setItems(r.items); setRetention(r.retention_days); })
      .catch((e) => { setItems([]); setError(e instanceof Error ? e.message : "Historique indisponible"); });
  }, []);

  const toggle = (id: string) => {
    if (openId === id) { setOpenId(null); return; }
    setOpenId(id);
    if (details[id]) return;
    getHistoryItem(id)
      .then((item) => setDetails((d) => ({ ...d, [id]: item })))
      .catch((e) => setDetails((d) => ({ ...d, [id]: { error: e instanceof Error ? e.message : "Lecture impossible" } })));
  };

  const remove = async (id: string) => {
    if (!window.confirm("Supprimer définitivement cet élément ?")) return;
    try {
      await deleteHistoryItem(id);
      setItems((list) => (list ?? []).filter((i) => i.id !== id));
      if (openId === id) setOpenId(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Suppression impossible");
    }
  };

  const removeAll = async () => {
    if (!window.confirm("Supprimer définitivement tout votre historique ?")) return;
    try {
      await clearHistory();
      setItems([]);
      setOpenId(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Suppression impossible");
    }
  };

  const copy = async (item: HistoryItem) => {
    await navigator.clipboard.writeText(historyAsText(item));
    setCopiedId(item.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const download = (item: HistoryItem) => {
    const url = URL.createObjectURL(new Blob([historyAsText(item)], { type: "text/plain;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${item.title.replace(/[^\p{L}\p{N}._-]+/gu, "_")}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <AppHeader />
      <main style={{ maxWidth: "720px", margin: "0 auto", padding: "40px 24px 64px" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: "16px", flexWrap: "wrap" }}>
          <h1 className="font-serif" style={{ fontSize: "clamp(26px, 4vw, 34px)", color: "var(--foreground)", margin: 0 }}>
            Mon historique
          </h1>
          {items && items.length > 0 && (
            <button onClick={removeAll} style={{ ...linkButton, color: "#e87070" }}>Tout supprimer</button>
          )}
        </div>
        <p style={{ fontSize: "13px", color: "var(--muted)", lineHeight: 1.6, margin: "12px 0 28px" }}>
          Vos traductions et comptes-rendus sont conservés {retention} jours, chiffrés, puis supprimés
          automatiquement. L&apos;audio n&apos;est jamais conservé.
        </p>

        {error && <p style={{ fontSize: "13px", color: "#e87070", marginBottom: "16px" }}>{error}</p>}

        {items === null && <p style={{ fontSize: "13px", color: "var(--muted)" }}>Chargement…</p>}

        {items && items.length === 0 && !error && (
          <div style={{ ...card, padding: "28px 24px", textAlign: "center" }}>
            <p style={{ fontSize: "14px", color: "var(--foreground)", margin: "0 0 8px" }}>Aucun traitement enregistré.</p>
            <p style={{ fontSize: "13px", color: "var(--muted)", margin: 0 }}>
              Vos prochains résultats apparaîtront ici.{" "}
              <Link href="/translate" style={{ color: "var(--accent)" }}>Traduire un audio</Link>
            </p>
          </div>
        )}

        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          {(items ?? []).map((meta) => {
            const open   = openId === meta.id;
            const detail = details[meta.id];
            return (
              <div key={meta.id} style={card}>
                <button onClick={() => toggle(meta.id)} aria-expanded={open} style={{
                  width: "100%", display: "flex", alignItems: "center", gap: "12px", padding: "14px 18px",
                  background: "none", border: "none", cursor: "pointer", textAlign: "left", color: "var(--foreground)",
                }}>
                  <span style={{ flex: 1, minWidth: 0 }}>
                    <span style={{ display: "block", fontSize: "14px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {meta.title}
                    </span>
                    <span style={{ display: "block", fontSize: "12px", color: "var(--muted)", marginTop: "3px" }}>
                      {[KIND_LABEL[meta.kind] ?? meta.kind, fmtDate(meta.created_at), fmtDuration(meta.duration_s),
                        meta.lang?.toUpperCase()].filter(Boolean).join(" · ")}
                    </span>
                  </span>
                  <span style={{ fontSize: "12px", color: "var(--accent)" }}>{open ? "Fermer" : "Ouvrir"}</span>
                </button>

                {open && (
                  <div style={{ padding: "4px 18px 18px", borderTop: "1px solid var(--border)" }}>
                    {!detail && <p style={{ fontSize: "13px", color: "var(--muted)", marginTop: "14px" }}>Chargement…</p>}
                    {detail && "error" in detail && (
                      <p style={{ fontSize: "13px", color: "#e87070", marginTop: "14px" }}>{detail.error}</p>
                    )}
                    {detail && !("error" in detail) && (
                      <>
                        <Content item={detail} />
                        <div style={{ display: "flex", gap: "18px", marginTop: "16px", flexWrap: "wrap" }}>
                          <button onClick={() => copy(detail)} style={linkButton}>
                            {copiedId === detail.id ? "Copié" : "Copier"}
                          </button>
                          <button onClick={() => download(detail)} style={linkButton}>Télécharger (.txt)</button>
                          <button onClick={() => remove(meta.id)} style={{ ...linkButton, color: "#e87070" }}>Supprimer</button>
                        </div>
                      </>
                    )}
                    {detail && "error" in detail && (
                      <button onClick={() => remove(meta.id)} style={{ ...linkButton, color: "#e87070", marginTop: "8px" }}>
                        Supprimer
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </main>
      <FooterMinimal />
    </>
  );
}

function Content({ item }: { item: HistoryItem }) {
  const c = item.content;
  if (isMeeting(c)) {
    return (
      <>
        <Block label="Compte-rendu" text={c.summary} />
        <details style={{ marginTop: "14px" }}>
          <summary style={{ fontSize: "12px", color: "var(--accent)", cursor: "pointer" }}>Transcription complète</summary>
          <p style={textStyle}>{c.transcript}</p>
        </details>
      </>
    );
  }
  if (c.segments?.length) {
    return (
      <div style={{ marginTop: "14px", display: "flex", flexDirection: "column", gap: "12px" }}>
        {c.segments.map((s, i) => (
          <div key={i}>
            <div style={labelStyle}>Intervenant {s.speaker}</div>
            <p style={{ ...textStyle, marginTop: "4px", color: "var(--muted)" }}>{s.original}</p>
            <p style={{ ...textStyle, marginTop: "4px" }}>{s.translation}</p>
          </div>
        ))}
      </div>
    );
  }
  return (
    <>
      <Block label="Traduction" text={c.translation} />
      <Block label="Texte d'origine" text={c.source_text} muted />
    </>
  );
}

function Block({ label, text, muted }: { label: string; text: string; muted?: boolean }) {
  return (
    <div style={{ marginTop: "14px" }}>
      <div style={labelStyle}>{label}</div>
      <p style={{ ...textStyle, color: muted ? "var(--muted)" : "var(--foreground)" }}>{text}</p>
    </div>
  );
}

const card: React.CSSProperties = {
  borderRadius: "14px", background: "var(--surface)", border: "1px solid var(--border)", overflow: "hidden",
};
const labelStyle: React.CSSProperties = {
  fontSize: "10px", letterSpacing: "0.2em", textTransform: "uppercase", color: "var(--muted)",
};
const textStyle: React.CSSProperties = {
  fontSize: "14px", lineHeight: 1.7, color: "var(--foreground)", margin: "8px 0 0",
  whiteSpace: "pre-wrap", overflowWrap: "anywhere",
};
const linkButton: React.CSSProperties = {
  background: "none", border: "none", padding: 0, font: "inherit", fontSize: "13px",
  color: "var(--accent)", cursor: "pointer",
};
