"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import * as Tabs from "@radix-ui/react-tabs";
import { getMe } from "@/lib/auth";
import {
  getAdminStats, getAdminUsers, getLangfuseMetrics, updateAdminUser, deleteAdminUser,
  getServicesHealth, getTrafficEvents, openTrafficStream, getExperiments, synthesizeTTS,
  type AdminUser, type AdminStats, type LangfuseMetrics, type LangfuseModelStat,
  type ServiceHealth, type TrafficEvent, type TrafficSnapshot,
  type ExperimentRun, type ExperimentsResponse,
} from "@/lib/admin";

// ── Palette ───────────────────────────────────────────────────────────────────
const C = {
  stt:    "#7eb8c9",
  llm:    "#c9a96e",
  tts:    "#9b7ec9",
  green:  "#7ec9a0",
  red:    "#e87070",
  muted:  "var(--muted)",
  accent: "var(--accent)",
  border: "var(--border)",
  surface:"var(--surface)",
  bg:     "var(--background)",
  fg:     "var(--foreground)",
};

// ── Tiny helpers ──────────────────────────────────────────────────────────────
function ms(v: number) {
  return v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${Math.round(v)}ms`;
}

function pct(v: number) { return `${Math.round(v * 100)}%`; }

// ── SVG Bar chart ─────────────────────────────────────────────────────────────
function BarChart({
  data, color = C.accent, height = 80,
}: {
  data: { label: string; value: number }[];
  color?: string;
  height?: number;
}) {
  if (!data.length) return null;
  const max   = Math.max(...data.map(d => d.value), 1);
  const W     = 100 / data.length;

  return (
    <svg width="100%" height={height + 28} style={{ overflow: "visible" }}>
      {data.map((d, i) => {
        const barH = (d.value / max) * height;
        const x    = i * W + W * 0.1;
        const w    = W * 0.8;
        return (
          <g key={i}>
            <rect
              x={`${x}%`} y={height - barH} width={`${w}%`} height={barH}
              rx="3" fill={color} opacity={0.8}
            />
            <text
              x={`${x + w / 2}%`} y={height + 14}
              textAnchor="middle" fill="var(--muted)"
              fontSize="9" fontFamily="monospace"
            >
              {d.label}
            </text>
            <text
              x={`${x + w / 2}%`} y={height - barH - 4}
              textAnchor="middle" fill={color}
              fontSize="9" fontFamily="monospace"
            >
              {d.value > 0 ? ms(d.value) : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ── SVG Histogram ─────────────────────────────────────────────────────────────
function Histogram({
  values, bins = 10, color = C.accent, height = 70, label = "",
}: {
  values: number[]; bins?: number; color?: string; height?: number; label?: string;
}) {
  if (!values.length) return <p style={{ fontSize: "12px", color: C.muted }}>Aucune donnée</p>;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const step = (max - min) / bins || 1;
  const counts = Array(bins).fill(0);
  values.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / step), bins - 1);
    counts[idx]++;
  });
  const maxC  = Math.max(...counts, 1);
  const W     = 100 / bins;

  return (
    <div>
      {label && <p style={{ fontSize: "11px", color: C.muted, marginBottom: "6px", letterSpacing: "0.1em" }}>{label}</p>}
      <svg width="100%" height={height + 24} style={{ overflow: "visible" }}>
        {counts.map((c, i) => {
          const barH = (c / maxC) * height;
          return (
            <g key={i}>
              <rect
                x={`${i * W + 0.5}%`} y={height - barH}
                width={`${W - 1}%`} height={barH}
                rx="2" fill={color} opacity={0.75}
              />
              {i % 2 === 0 && (
                <text
                  x={`${i * W + W / 2}%`} y={height + 14}
                  textAnchor="middle" fill="var(--muted)"
                  fontSize="8" fontFamily="monospace"
                >
                  {(min + i * step).toFixed(1)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, color, sub }: { label: string; value: string | number; color?: string; sub?: string }) {
  return (
    <div style={{
      padding: "18px 20px", borderRadius: "16px",
      background: C.surface, border: `1px solid ${C.border}`,
      display: "flex", flexDirection: "column", gap: "6px",
    }}>
      <span style={{ fontSize: "10px", letterSpacing: "0.2em", textTransform: "uppercase", color: C.muted }}>{label}</span>
      <span style={{ fontSize: "26px", fontWeight: 600, color: color ?? C.fg, fontVariantNumeric: "tabular-nums" }}>{value}</span>
      {sub && <span style={{ fontSize: "11px", color: C.muted }}>{sub}</span>}
    </div>
  );
}

// ── Coming soon stub ──────────────────────────────────────────────────────────
function ComingSoon({ title, description, icon }: { title: string; description: string; icon: React.ReactNode }) {
  return (
    <div style={{
      textAlign: "center", padding: "60px 24px",
      border: `1px dashed ${C.border}`, borderRadius: "20px",
    }}>
      <div style={{ color: C.muted, opacity: 0.4, marginBottom: "16px" }}>{icon}</div>
      <p style={{ fontSize: "14px", color: C.muted, marginBottom: "8px" }}>{title}</p>
      <p style={{ fontSize: "12px", color: C.muted, opacity: 0.6, maxWidth: "36ch", margin: "0 auto" }}>{description}</p>
    </div>
  );
}

// ── Overview tab ──────────────────────────────────────────────────────────────
function OverviewTab({ stats, langfuse }: { stats: AdminStats | null; langfuse: LangfuseMetrics | null }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>

      {/* User stats */}
      <section>
        <SectionTitle>Utilisateurs</SectionTitle>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: "12px" }}>
          <StatCard label="Total"   value={stats?.total_users  ?? "—"} color={C.fg} />
          <StatCard label="Actifs"  value={stats?.active_users ?? "—"} color={C.green} />
          <StatCard label="Admins"  value={stats?.admin_users  ?? "—"} color={C.accent} />
        </div>
      </section>

      {/* Pipeline stats from Langfuse */}
      {langfuse?.connected && (
        <section>
          <SectionTitle>Pipeline — aperçu global</SectionTitle>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: "12px" }}>
            <StatCard label="Traductions"  value={langfuse.total_traces} />
            <StatCard label="Latence moy." value={ms(langfuse.avg_total_ms)}    color={C.muted} />
            <StatCard label="STT moy."     value={ms(langfuse.avg_stt_ms)}      color={C.stt} />
            <StatCard label="LLM moy."     value={ms(langfuse.avg_llm_ms)}      color={C.llm} />
            <StatCard label="Conf. langue" value={pct(langfuse.avg_language_prob)} color={C.green} />
            <StatCard label="BLEU moy."   value={langfuse.avg_bleu   > 0 ? langfuse.avg_bleu.toFixed(3)   : "—"} color={C.tts} />
            <StatCard label="METEOR moy." value={(langfuse.avg_meteor ?? 0) > 0 ? (langfuse.avg_meteor ?? 0).toFixed(4) : "—"} color={C.green} />
            <StatCard label="WER moy."    value={(langfuse.avg_wer    ?? 0) > 0 ? (langfuse.avg_wer    ?? 0).toFixed(4) : "—"} color={C.red} sub="↓ mieux" />
            <StatCard
              label="Coût total"
              value={(langfuse.total_cost_usd ?? 0) > 0 ? `$${(langfuse.total_cost_usd ?? 0).toFixed(4)}` : "—"}
              color={C.accent}
              sub={(langfuse.total_tokens ?? 0) > 0 ? `${(langfuse.total_tokens ?? 0).toLocaleString()} tokens` : undefined}
            />
            <StatCard
              label="Coût moy. / run"
              value={(langfuse.avg_cost_usd ?? 0) > 0 ? `$${(langfuse.avg_cost_usd ?? 0).toFixed(5)}` : "—"}
              color={C.accent}
              sub={(langfuse.avg_tokens ?? 0) > 0 ? `${Math.round(langfuse.avg_tokens ?? 0)} tok/run` : undefined}
            />
          </div>
        </section>
      )}

      {langfuse && !langfuse.connected && (
        <div style={{
          padding: "16px 20px", borderRadius: "12px", fontSize: "13px",
          background: "rgba(232,112,112,0.06)", border: "1px solid rgba(232,112,112,0.2)", color: C.red,
        }}>
          Langfuse non connecté — {langfuse.error}
        </div>
      )}
    </div>
  );
}

// ── Traces & Models tab ───────────────────────────────────────────────────────
type SortKey = "count" | "avg_stt_ms" | "avg_llm_ms" | "avg_total_ms" | "avg_bleu" | "avg_meteor" | "avg_wer" | "avg_cost_usd";

const COLUMNS: { key: SortKey | null; label: string; color?: string; asc?: boolean }[] = [
  { key: null,           label: "Whisper",     color: C.stt },
  { key: null,           label: "LLM" },
  { key: null,           label: "Prompt" },
  { key: "count",        label: "Runs",        asc: false },
  { key: "avg_stt_ms",   label: "STT moy.",    color: C.stt,   asc: true },
  { key: "avg_llm_ms",   label: "LLM moy.",    color: C.llm,   asc: true },
  { key: "avg_total_ms", label: "Total moy.",  asc: true },
  { key: "avg_bleu",     label: "BLEU ↑",      color: C.tts,   asc: false },
  { key: "avg_meteor",   label: "METEOR ↑",    color: C.green, asc: false },
  { key: "avg_wer",      label: "WER ↓",       color: C.red,   asc: true },
  { key: "avg_cost_usd", label: "Coût ↓",      color: C.accent, asc: true },
];

function TracesTab({ langfuse }: { langfuse: LangfuseMetrics | null }) {
  const [sortKey, setSortKey]   = useState<SortKey>("count");
  const [sortAsc, setSortAsc]   = useState(false);

  if (!langfuse) return <Loader />;

  if (!langfuse.connected) {
    return (
      <div style={{
        padding: "16px 20px", borderRadius: "12px", fontSize: "13px",
        background: "rgba(232,112,112,0.06)", border: "1px solid rgba(232,112,112,0.2)", color: C.red,
      }}>
        Langfuse non connecté — {langfuse.error}<br />
        <span style={{ fontSize: "12px", opacity: 0.8 }}>
          Configurez LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY dans votre .env puis relancez le gateway.
        </span>
      </div>
    );
  }

  const latencyData = [
    { label: "STT",   value: langfuse.avg_stt_ms },
    { label: "LLM",   value: langfuse.avg_llm_ms },
    { label: "Total", value: langfuse.avg_total_ms },
  ];

  const sortedStats = [...langfuse.model_stats].sort((a, b) => {
    const va = (a[sortKey] ?? 0) as number;
    const vb = (b[sortKey] ?? 0) as number;
    return sortAsc ? va - vb : vb - va;
  });

  function handleSort(col: typeof COLUMNS[0]) {
    if (!col.key) return;
    if (sortKey === col.key) {
      setSortAsc(prev => !prev);
    } else {
      setSortKey(col.key);
      setSortAsc(col.asc ?? false);
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>

      {/* Latency bar chart */}
      <Card title="Latences moyennes par service">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "20px", alignItems: "end" }}>
          {latencyData.map(d => (
            <div key={d.label} style={{ textAlign: "center" }}>
              <div style={{ height: "80px", display: "flex", alignItems: "flex-end", justifyContent: "center" }}>
                <div style={{
                  width: "48px",
                  height: `${Math.round((d.value / Math.max(langfuse.avg_total_ms, 1)) * 80)}px`,
                  borderRadius: "6px 6px 0 0",
                  background: d.label === "STT" ? C.stt : d.label === "LLM" ? C.llm : C.muted,
                  opacity: 0.85, minHeight: "4px", transition: "height 0.6s ease",
                }} />
              </div>
              <p style={{ fontSize: "16px", fontWeight: 600, marginTop: "8px", fontVariantNumeric: "tabular-nums",
                color: d.label === "STT" ? C.stt : d.label === "LLM" ? C.llm : C.muted }}>
                {ms(d.value)}
              </p>
              <p style={{ fontSize: "10px", color: C.muted, letterSpacing: "0.15em" }}>{d.label}</p>
            </div>
          ))}
        </div>
      </Card>

      {/* Distribution histograms */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
        <Card title="Distribution confiance langue">
          <Histogram values={langfuse.language_probs} bins={10} color={C.green} height={60} />
          <p style={{ fontSize: "11px", color: C.muted, marginTop: "8px" }}>
            Moy. {pct(langfuse.avg_language_prob)} · {langfuse.language_probs.length} traces
          </p>
        </Card>
        <Card title="Distribution BLEU">
          {langfuse.bleu_scores.length > 0 ? (
            <>
              <Histogram values={langfuse.bleu_scores} bins={8} color={C.tts} height={60} />
              <p style={{ fontSize: "11px", color: C.muted, marginTop: "8px" }}>
                Moy. {langfuse.avg_bleu.toFixed(3)} · {langfuse.bleu_scores.length} traces · ↑ mieux
              </p>
            </>
          ) : (
            <p style={{ fontSize: "12px", color: C.muted }}>Aucun score BLEU enregistré</p>
          )}
        </Card>
        <Card title="Distribution METEOR">
          {(langfuse.meteor_scores ?? []).length > 0 ? (
            <>
              <Histogram values={langfuse.meteor_scores} bins={8} color={C.green} height={60} />
              <p style={{ fontSize: "11px", color: C.muted, marginTop: "8px" }}>
                Moy. {(langfuse.avg_meteor ?? 0).toFixed(4)} · {langfuse.meteor_scores.length} traces · ↑ mieux
              </p>
            </>
          ) : (
            <p style={{ fontSize: "12px", color: C.muted }}>Aucun score METEOR — relancez eval_golden.py</p>
          )}
        </Card>
        <Card title="Distribution WER (STT)">
          {(langfuse.wer_scores ?? []).length > 0 ? (
            <>
              <Histogram values={langfuse.wer_scores} bins={8} color={C.red} height={60} />
              <p style={{ fontSize: "11px", color: C.muted, marginTop: "8px" }}>
                Moy. {(langfuse.avg_wer ?? 0).toFixed(4)} · {langfuse.wer_scores.length} traces · ↓ mieux
              </p>
            </>
          ) : (
            <p style={{ fontSize: "12px", color: C.muted }}>Aucun score WER — créez les refs FR dans data/golden/references/</p>
          )}
        </Card>
      </div>

      {/* Model comparison table with sortable columns */}
      {langfuse.model_stats.length > 0 && (
        <Card title="Comparaison des modèles">
          <p style={{ fontSize: "11px", color: C.muted, marginBottom: "12px" }}>
            Cliquez sur un en-tête pour trier · {sortKey} {sortAsc ? "↑" : "↓"}
          </p>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
              <thead>
                <tr>
                  {COLUMNS.map(col => (
                    <th key={col.label}
                      onClick={() => handleSort(col)}
                      style={{
                        padding: "8px 12px", textAlign: "left", fontWeight: 600,
                        letterSpacing: "0.08em", fontSize: "10px",
                        borderBottom: `1px solid ${C.border}`,
                        color: sortKey === col.key ? (col.color ?? C.accent) : C.muted,
                        cursor: col.key ? "pointer" : "default",
                        userSelect: "none",
                        whiteSpace: "nowrap",
                      }}>
                      {col.label}
                      {sortKey === col.key && <span style={{ marginLeft: "4px" }}>{sortAsc ? "↑" : "↓"}</span>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedStats.map((m, i) => (
                  <tr key={i} style={{
                    borderBottom: `1px solid ${C.border}`,
                    background: i === 0 && sortKey === "avg_bleu" ? "rgba(155,126,201,0.04)" : "none",
                  }}>
                    <td style={{ padding: "10px 12px", color: C.stt }}>{m.whisper}</td>
                    <td style={{ padding: "10px 12px", color: C.llm, maxWidth: "160px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {m.llm.replace("groq/", "")}
                    </td>
                    <td style={{ padding: "10px 12px", color: C.muted }}>{m.prompt_version}</td>
                    <td style={{ padding: "10px 12px", color: C.fg, fontWeight: 600 }}>{m.count}</td>
                    <td style={{ padding: "10px 12px", color: C.stt, fontVariantNumeric: "tabular-nums" }}>{ms(m.avg_stt_ms)}</td>
                    <td style={{ padding: "10px 12px", color: C.llm, fontVariantNumeric: "tabular-nums" }}>{ms(m.avg_llm_ms)}</td>
                    <td style={{ padding: "10px 12px", color: C.muted, fontVariantNumeric: "tabular-nums" }}>{ms(m.avg_total_ms)}</td>
                    <td style={{ padding: "10px 12px", color: C.tts, fontVariantNumeric: "tabular-nums" }}>
                      {m.avg_bleu != null ? m.avg_bleu.toFixed(3) : <span style={{ opacity: 0.4 }}>—</span>}
                    </td>
                    <td style={{ padding: "10px 12px", color: C.green, fontVariantNumeric: "tabular-nums" }}>
                      {m.avg_meteor != null ? m.avg_meteor.toFixed(4) : <span style={{ opacity: 0.4 }}>—</span>}
                    </td>
                    <td style={{ padding: "10px 12px", color: C.red, fontVariantNumeric: "tabular-nums" }}>
                      {m.avg_wer != null ? m.avg_wer.toFixed(4) : <span style={{ opacity: 0.4 }}>—</span>}
                    </td>
                    <td style={{ padding: "10px 12px", color: C.accent, fontVariantNumeric: "tabular-nums" }}>
                      {m.avg_cost_usd != null && m.avg_cost_usd > 0
                        ? `$${m.avg_cost_usd.toFixed(5)}`
                        : <span style={{ opacity: 0.4 }}>—</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

// ── Experiments tab ───────────────────────────────────────────────────────────
type ExpSortKey = "bleu" | "meteor" | "wer" | "tts_wer" | "latency_stt_ms" | "latency_llm_ms" | "latency_total_ms";

const EXP_COLS: { key: ExpSortKey | null; label: string; color?: string }[] = [
  { key: null,               label: "Audio" },
  { key: null,               label: "Whisper" },
  { key: null,               label: "LLM" },
  { key: null,               label: "Prompt" },
  { key: "bleu",             label: "BLEU",    color: C.tts },
  { key: "meteor",           label: "METEOR",  color: C.green },
  { key: "wer",              label: "WER ↓",    color: C.red },
  { key: "tts_wer",          label: "TTS WER ↓", color: C.tts },
  { key: "latency_stt_ms",   label: "STT ms",  color: C.stt },
  { key: "latency_llm_ms",   label: "LLM ms",  color: C.llm },
  { key: "latency_total_ms", label: "Total ms" },
];

function ExperimentsTab() {
  const [data,    setData]    = useState<ExperimentsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState<string | null>(null);

  const [filterWhisper, setFilterWhisper] = useState("all");
  const [filterLLM,     setFilterLLM]     = useState("all");
  const [filterPrompt,  setFilterPrompt]  = useState("all");

  const [sortKey, setSortKey] = useState<ExpSortKey>("bleu");
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => {
    getExperiments()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p style={{ color: C.muted, fontSize: "13px" }}>Chargement…</p>;
  if (error)   return <p style={{ color: "#e87070", fontSize: "13px" }}>{error}</p>;
  if (!data?.csv_exists) return (
    <div style={{ color: C.muted, fontSize: "13px", padding: "24px", textAlign: "center" }}>
      CSV introuvable — lance <code>python scripts/eval_golden.py</code>
    </div>
  );

  const whispers = ["all", ...Array.from(new Set(data.runs.map(r => r.whisper_model)))];
  const llms     = ["all", ...Array.from(new Set(data.runs.map(r => r.llm_model.replace("groq/", ""))))];
  const prompts  = ["all", ...Array.from(new Set(data.runs.map(r => r.prompt_version)))];

  const filtered = data.runs
    .filter(r =>
      (filterWhisper === "all" || r.whisper_model === filterWhisper) &&
      (filterLLM     === "all" || r.llm_model.replace("groq/", "") === filterLLM) &&
      (filterPrompt  === "all" || r.prompt_version === filterPrompt)
    )
    .sort((a, b) => {
      const va = (a[sortKey] ?? -1) as number;
      const vb = (b[sortKey] ?? -1) as number;
      return sortAsc ? va - vb : vb - va;
    });

  const withBleu   = filtered.filter(r => r.bleu   != null);
  const withMeteor = filtered.filter(r => r.meteor != null);
  const withWer    = filtered.filter(r => r.wer    != null);
  const avg = (arr: number[]) => arr.length ? arr.reduce((a,b) => a+b, 0) / arr.length : null;
  const fmt = (v: number | null, d = 2) => v != null ? v.toFixed(d) : "—";

  const selectStyle: React.CSSProperties = {
    background: C.surface, border: `1px solid ${C.border}`, borderRadius: "8px",
    color: C.fg, fontSize: "12px", padding: "5px 10px", cursor: "pointer",
  };

  function handleSort(col: typeof EXP_COLS[0]) {
    if (!col.key) return;
    if (sortKey === col.key) setSortAsc(v => !v);
    else { setSortKey(col.key); setSortAsc(false); }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>

      {/* KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
        <StatCard label="Runs"        value={String(filtered.length)} sub={`sur ${data.total} total`} />
        <StatCard label="BLEU moy."   value={fmt(avg(withBleu.map(r => r.bleu!)))}        sub="sacrebleu" color={C.tts} />
        <StatCard label="METEOR moy." value={fmt(avg(withMeteor.map(r => r.meteor!)), 4)} sub="nltk"      color={C.green} />
        <StatCard label="WER moy."    value={fmt(avg(withWer.map(r => r.wer!)), 4)}        sub="↓ mieux"   color={C.red} />
      </div>

      {/* Filtres */}
      <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "center" }}>
        <span style={{ fontSize: "11px", color: C.muted, letterSpacing: "0.1em" }}>FILTRER</span>
        <select style={selectStyle} value={filterWhisper} onChange={e => setFilterWhisper(e.target.value)}>
          {whispers.map(w => <option key={w} value={w}>{w === "all" ? "Whisper (tous)" : w}</option>)}
        </select>
        <select style={selectStyle} value={filterLLM} onChange={e => setFilterLLM(e.target.value)}>
          {llms.map(l => <option key={l} value={l}>{l === "all" ? "LLM (tous)" : l}</option>)}
        </select>
        <select style={selectStyle} value={filterPrompt} onChange={e => setFilterPrompt(e.target.value)}>
          {prompts.map(p => <option key={p} value={p}>{p === "all" ? "Prompt (tous)" : p}</option>)}
        </select>
        <span style={{ fontSize: "11px", color: C.muted, marginLeft: "auto" }}>
          {filtered.length} résultats · tri {sortKey} {sortAsc ? "↑" : "↓"}
        </span>
      </div>

      {/* Tableau */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
          <thead>
            <tr style={{ borderBottom: `1px solid ${C.border}` }}>
              {EXP_COLS.map(col => (
                <th
                  key={col.label}
                  onClick={() => handleSort(col)}
                  style={{
                    padding: "8px 12px", textAlign: "left", fontWeight: 600,
                    fontSize: "11px", letterSpacing: "0.08em",
                    color: sortKey === col.key ? (col.color ?? C.accent) : C.muted,
                    cursor: col.key ? "pointer" : "default",
                    whiteSpace: "nowrap", userSelect: "none",
                  }}
                >
                  {col.label}
                  {sortKey === col.key && <span style={{ marginLeft: "4px" }}>{sortAsc ? "↑" : "↓"}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => (
              <tr key={r.run_id} style={{
                borderBottom: `1px solid ${C.border}20`,
                background: i % 2 === 0 ? "transparent" : `${C.surface}60`,
              }}>
                <td style={{ padding: "9px 12px", color: C.muted, maxWidth: "140px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {r.audio.replace(/\.mp3$/, "").replace("flash_", "").replace(/_/g, " ")}
                </td>
                <td style={{ padding: "9px 12px", color: C.stt }}>{r.whisper_model}</td>
                <td style={{ padding: "9px 12px", color: C.llm }}>{r.llm_model.replace("groq/", "")}</td>
                <td style={{ padding: "9px 12px", color: C.muted }}>{r.prompt_version}</td>
                <td style={{ padding: "9px 12px", color: C.tts,   fontWeight: 600 }}>{fmt(r.bleu)}</td>
                <td style={{ padding: "9px 12px", color: C.green, fontWeight: 600 }}>{fmt(r.meteor, 4)}</td>
                <td style={{ padding: "9px 12px", color: C.red,   fontWeight: 600 }}>{fmt(r.wer, 4)}</td>
                <td style={{ padding: "9px 12px", color: C.tts,   fontWeight: 600 }}>{fmt(r.tts_wer, 4)}</td>
                <td style={{ padding: "9px 12px", color: C.stt }}>{r.latency_stt_ms != null ? `${Math.round(r.latency_stt_ms)}` : "—"}</td>
                <td style={{ padding: "9px 12px", color: C.llm }}>{r.latency_llm_ms != null ? `${Math.round(r.latency_llm_ms)}` : "—"}</td>
                <td style={{ padding: "9px 12px", color: C.muted }}>{r.latency_total_ms != null ? `${Math.round(r.latency_total_ms)}` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Roadmap MLflow */}
      <div style={{
        padding: "12px 16px", borderRadius: "10px", fontSize: "11px",
        background: "rgba(201,169,110,0.06)", border: `1px solid ${C.border}`, color: C.muted,
      }}>
        Phase 4 — intégration MLflow model registry prévue
      </div>
    </div>
  );
}

// ── Infrastructure tab ────────────────────────────────────────────────────────
function InfraTab() {
  const [services, setServices] = useState<ServiceHealth[]>([]);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState<string | null>(null);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getServicesHealth();
      setServices(data.services ?? []);
      setLastCheck(new Date());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Erreur inconnue");
    } finally {
      setLoading(false);
    }
  }, []);

  // Petit délai pour laisser le token s'initialiser
  useEffect(() => { setTimeout(() => refresh(), 300); }, []);

  const upCount   = services.filter(s => s.status === "up").length;
  const downCount = services.filter(s => s.status !== "up").length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>

      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", gap: "10px" }}>
          {upCount > 0 && (
            <span style={{
              fontSize: "11px", padding: "3px 10px", borderRadius: "999px",
              background: "rgba(126,201,160,0.1)", color: C.green, letterSpacing: "0.1em",
            }}>{upCount} en ligne</span>
          )}
          {downCount > 0 && (
            <span style={{
              fontSize: "11px", padding: "3px 10px", borderRadius: "999px",
              background: "rgba(232,112,112,0.08)", color: C.red, letterSpacing: "0.1em",
            }}>{downCount} hors ligne</span>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          {lastCheck && (
            <span style={{ fontSize: "11px", color: C.muted, opacity: 0.6 }}>
              {lastCheck.toLocaleTimeString("fr-FR")}
            </span>
          )}
          <button onClick={refresh} disabled={loading} style={{
            padding: "5px 12px", borderRadius: "8px", fontSize: "11px",
            cursor: loading ? "wait" : "pointer",
            background: "none", border: `1px solid ${C.border}`, color: C.muted,
          }}>
            {loading ? "…" : "↻ Actualiser"}
          </button>
        </div>
      </div>

      {/* Service cards */}
      {error && (
        <div style={{
          padding: "12px 16px", borderRadius: "10px", fontSize: "12px",
          background: "rgba(232,112,112,0.06)", border: "1px solid rgba(232,112,112,0.2)", color: C.red,
        }}>
          Impossible de contacter le gateway — {error}
        </div>
      )}
      {loading && services.length === 0 ? <Loader /> : (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
          {services.map(s => {
            const isUp     = s.status === "up";
            const dotColor = isUp ? C.green : C.red;
            return (
              <div key={s.name} style={{
                padding: "16px 18px", borderRadius: "14px",
                background: C.surface,
                border: `1px solid ${isUp ? C.border : "rgba(232,112,112,0.25)"}`,
                display: "flex", alignItems: "center", gap: "12px",
                transition: "border-color 0.3s",
              }}>
                <div style={{
                  width: "9px", height: "9px", borderRadius: "50%", flexShrink: 0,
                  background: dotColor,
                  boxShadow: isUp ? `0 0 6px ${dotColor}` : "none",
                  animation: isUp ? "pulse-dot 2s ease-in-out infinite" : "none",
                }} />
                <div style={{ flex: 1 }}>
                  <p style={{ fontSize: "13px", fontWeight: 500, color: C.fg }}>{s.name}</p>
                  <p style={{ fontSize: "11px", color: C.muted }}>:{s.port}</p>
                </div>
                <div style={{ textAlign: "right" }}>
                  <span style={{
                    fontSize: "10px", padding: "2px 8px", borderRadius: "999px",
                    background: isUp ? "rgba(126,201,160,0.1)" : "rgba(232,112,112,0.08)",
                    color: isUp ? C.green : C.red, letterSpacing: "0.1em", display: "block",
                    marginBottom: "4px",
                  }}>
                    {s.status}
                  </span>
                  {isUp && (
                    <span style={{ fontSize: "10px", color: C.muted, fontVariantNumeric: "tabular-nums" }}>
                      {s.latency_ms}ms
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Grafana embedded — live metrics from Prometheus */}
      <Card title="Grafana — métriques live (Prometheus)">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "10px" }}>
          <p style={{ fontSize: "11px", color: C.muted }}>
            Requêtes/s, latence p95, taux erreurs, services up
          </p>
          <a href="http://localhost:3001/d/llmops-overview" target="_blank" rel="noreferrer" style={{
            fontSize: "11px", padding: "4px 12px", borderRadius: "8px",
            background: "rgba(201,169,110,0.08)", border: `1px solid ${C.border}`,
            color: C.accent, textDecoration: "none",
          }}>
            Ouvrir Grafana ↗
          </a>
        </div>
        <iframe
          src="http://localhost:3001/d/llmops-overview?orgId=1&kiosk=tv&refresh=10s&theme=dark"
          style={{ width: "100%", height: "600px", border: `1px solid ${C.border}`, borderRadius: "12px" }}
          title="Grafana — LLMOps Overview"
        />
      </Card>
    </div>
  );
}

// ── Pipelines tab (Airflow stub) ──────────────────────────────────────────────
function PipelinesTab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div style={{
        padding: "14px 18px", borderRadius: "12px", fontSize: "12px",
        background: "rgba(201,169,110,0.06)", border: "1px solid var(--accent-dim)", color: C.accent,
        letterSpacing: "0.05em",
      }}>
        Phase finale — orchestration avec Apache Airflow
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
        {[
          { dag: "translate_pipeline", schedule: "@on_demand", runs: "84", status: "success" },
          { dag: "model_evaluation",   schedule: "@weekly",     runs: "—",  status: "pending" },
          { dag: "data_export",        schedule: "@daily",      runs: "—",  status: "pending" },
        ].map(d => (
          <div key={d.dag} style={{
            padding: "14px 18px", borderRadius: "14px",
            background: C.surface, border: `1px solid ${C.border}`,
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
              <span style={{ fontSize: "12px", fontWeight: 500, color: C.fg }}>{d.dag}</span>
              <span style={{
                fontSize: "10px", padding: "2px 8px", borderRadius: "999px",
                background: d.status === "success" ? "rgba(126,201,160,0.1)" : "rgba(201,169,110,0.08)",
                color: d.status === "success" ? C.green : C.accent, letterSpacing: "0.1em",
              }}>{d.status}</span>
            </div>
            <p style={{ fontSize: "11px", color: C.muted }}>{d.schedule} · {d.runs} runs</p>
          </div>
        ))}
      </div>

      <ComingSoon
        title="Airflow — Orchestration"
        description="DAGs d'entraînement, d'évaluation et d'export. Planification automatique et monitoring des pipelines ML."
        icon={
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
          </svg>
        }
      />
    </div>
  );
}

// ── Users tab ─────────────────────────────────────────────────────────────────
function UsersTab({ users, currentUserId, onRefresh }: {
  users: AdminUser[]; currentUserId: number; onRefresh: () => void;
}) {
  const [loading, setLoading]   = useState<number | null>(null);
  const [confirm, setConfirm]   = useState<number | null>(null);

  const toggle = async (u: AdminUser, field: "is_active" | "is_admin") => {
    setLoading(u.id);
    try { await updateAdminUser(u.id, { [field]: !u[field] }); onRefresh(); }
    finally { setLoading(null); }
  };

  const remove = async (id: number) => {
    setLoading(id);
    try { await deleteAdminUser(id); onRefresh(); }
    finally { setLoading(null); setConfirm(null); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      {users.map(u => (
        <div key={u.id} style={{
          display: "flex", alignItems: "center", gap: "12px",
          padding: "12px 16px", borderRadius: "14px",
          background: C.surface, border: `1px solid ${C.border}`,
          flexWrap: "wrap",
        }}>
          {/* Avatar */}
          <div style={{
            width: "32px", height: "32px", borderRadius: "50%", flexShrink: 0,
            background: u.is_admin ? "rgba(201,169,110,0.15)" : "rgba(255,255,255,0.05)",
            border: `1px solid ${u.is_admin ? "var(--accent-dim)" : C.border}`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "12px", fontWeight: 700,
            color: u.is_admin ? C.accent : C.muted,
          }}>
            {u.email[0].toUpperCase()}
          </div>

          {/* Email */}
          <div style={{ flex: 1, minWidth: "160px" }}>
            <p style={{ fontSize: "13px", color: C.fg }}>{u.email}</p>
            <p style={{ fontSize: "11px", color: C.muted }}>
              {u.created_at ? new Date(u.created_at).toLocaleDateString("fr-FR") : "—"}
            </p>
          </div>

          {/* Badges */}
          <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
            <span style={{
              fontSize: "10px", padding: "2px 8px", borderRadius: "999px", letterSpacing: "0.1em",
              background: u.is_active ? "rgba(126,201,160,0.1)" : "rgba(232,112,112,0.08)",
              color: u.is_active ? C.green : C.red,
            }}>
              {u.is_active ? "actif" : "inactif"}
            </span>
            {u.is_admin && (
              <span style={{
                fontSize: "10px", padding: "2px 8px", borderRadius: "999px", letterSpacing: "0.1em",
                background: "rgba(201,169,110,0.1)", color: C.accent,
              }}>
                admin
              </span>
            )}
          </div>

          {/* Actions (not self) */}
          {u.id !== currentUserId && (
            <div style={{ display: "flex", gap: "6px", marginLeft: "auto" }}>
              <ActionBtn
                label={u.is_active ? "Désactiver" : "Activer"}
                color={u.is_active ? C.red : C.green}
                loading={loading === u.id}
                onClick={() => toggle(u, "is_active")}
              />
              <ActionBtn
                label={u.is_admin ? "Rétrograder" : "Promouvoir"}
                color={C.accent}
                loading={loading === u.id}
                onClick={() => toggle(u, "is_admin")}
              />
              {confirm === u.id ? (
                <ActionBtn label="Confirmer ?" color={C.red} loading={loading === u.id} onClick={() => remove(u.id)} />
              ) : (
                <ActionBtn label="Supprimer" color={C.red} loading={false} onClick={() => setConfirm(u.id)} />
              )}
            </div>
          )}
          {u.id === currentUserId && (
            <span style={{ fontSize: "11px", color: C.muted, marginLeft: "auto", opacity: 0.5 }}>vous</span>
          )}
        </div>
      ))}
    </div>
  );
}

function ActionBtn({ label, color, loading, onClick }: {
  label: string; color: string; loading: boolean; onClick: () => void;
}) {
  return (
    <button onClick={onClick} disabled={loading} style={{
      padding: "5px 10px", borderRadius: "8px", fontSize: "11px", cursor: loading ? "wait" : "pointer",
      background: "none", border: `1px solid ${color}22`, color,
      transition: "background 0.15s",
    }}
    onMouseEnter={e => (e.currentTarget.style.background = `${color}12`)}
    onMouseLeave={e => (e.currentTarget.style.background = "none")}
    >
      {loading ? "…" : label}
    </button>
  );
}

// ── Shared layout helpers ─────────────────────────────────────────────────────
function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <p style={{
      fontSize: "10px", letterSpacing: "0.25em", textTransform: "uppercase",
      color: C.muted, marginBottom: "12px", fontWeight: 600,
    }}>
      {children}
    </p>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ borderRadius: "16px", background: C.surface, border: `1px solid ${C.border}`, overflow: "hidden" }}>
      <div style={{ padding: "12px 18px", borderBottom: `1px solid ${C.border}` }}>
        <span style={{ fontSize: "10px", letterSpacing: "0.2em", textTransform: "uppercase", color: C.muted, fontWeight: 600 }}>
          {title}
        </span>
      </div>
      <div style={{ padding: "18px" }}>{children}</div>
    </div>
  );
}

function Loader() {
  return (
    <div style={{ display: "flex", justifyContent: "center", padding: "48px" }}>
      <div style={{ display: "flex", gap: "6px" }}>
        {[0, 0.15, 0.3].map((d, i) => (
          <div key={i} style={{
            width: "8px", height: "8px", borderRadius: "50%",
            background: C.accent, animation: `pulse 1.2s ease-in-out ${d}s infinite`,
          }} />
        ))}
      </div>
    </div>
  );
}

// ── Traffic Live tab ──────────────────────────────────────────────────────────
const SEVERITY_COLOR: Record<string, string> = {
  high:   "#e87070",
  medium: "#c9a96e",
  low:    "#7eb8c9",
};
const SEVERITY_LABEL: Record<string, string> = {
  high:   "URGENT",
  medium: "ALERTE",
  low:    "INFO",
};
function TrafficTypeIcon({ type, color }: { type: string; color: string }) {
  const s: React.CSSProperties = { display: "inline-block", verticalAlign: "middle", width: 14, height: 14, flexShrink: 0 };
  switch (type) {
    case "accident": return (
      <svg viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={s}>
        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
    );
    case "fermeture": return (
      <svg viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" style={s}>
        <circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>
      </svg>
    );
    case "bouchon": return (
      <svg viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
        <rect x="1" y="8" width="22" height="8" rx="2"/><path d="M5 8V6a2 2 0 012-2h10a2 2 0 012 2v2"/><line x1="12" y1="12" x2="12.01" y2="12"/>
      </svg>
    );
    case "animal": return (
      <svg viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
        <circle cx="12" cy="12" r="2"/><path d="M6 6l2 4M18 6l-2 4M6 18l2-4M18 18l-2-4"/>
      </svg>
    );
    case "intemperies": return (
      <svg viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
        <path d="M20 17.58A5 5 0 0018 8h-1.26A8 8 0 104 16.25"/><line x1="8" y1="16" x2="8.01" y2="16"/><line x1="8" y1="20" x2="8.01" y2="20"/><line x1="12" y1="18" x2="12.01" y2="18"/><line x1="12" y1="22" x2="12.01" y2="22"/><line x1="16" y1="16" x2="16.01" y2="16"/><line x1="16" y1="20" x2="16.01" y2="20"/>
      </svg>
    );
    case "ralentissement": return (
      <svg viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
        <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
      </svg>
    );
    case "travaux": return (
      <svg viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
        <path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/>
      </svg>
    );
    case "vehicule_panne": return (
      <svg viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
        <rect x="1" y="8" width="22" height="8" rx="2"/><path d="M5 8V6a2 2 0 012-2h10a2 2 0 012 2v2"/><circle cx="7" cy="16" r="1"/><circle cx="17" cy="16" r="1"/>
      </svg>
    );
    default: return (
      <svg viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" style={s}>
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
    );
  }
}
const TYPE_LABEL: Record<string, string> = {
  accident:       "Accident",
  fermeture:      "Fermeture",
  bouchon:        "Bouchon",
  animal:         "Animal",
  intemperies:    "Intempéries",
  ralentissement: "Ralentissement",
  travaux:        "Travaux",
  vehicule_panne: "Véhicule en panne",
};

function cleanHint(hint: string): string {
  // Supprimer les "..." de début/fin et nettoyer les espaces
  return hint.replace(/^\.{2,}\s*/, "").replace(/\s*\.{2,}$/, "").trim();
}

function cleanDirection(dir: string): string {
  // Supprimer les mots parasites en fin ("et", "ou", "à", virgules...)
  return dir.replace(/\s+(et|ou|à|en|,|;)$/i, "").trim();
}

const LANG_LABELS: Record<string, string> = {
  fr: "FR", en: "EN", uk: "UK", es: "ES", de: "DE",
};
const LANG_FLAGS: Record<string, string> = {
  fr: "🇫🇷", en: "🇬🇧", uk: "🇺🇦", es: "🇪🇸", de: "🇩🇪",
};

// Singleton audio global — garantit une seule lecture à la fois sur toute la page
const _currentAudio: { el: HTMLAudioElement | null; setter: ((v: boolean) => void) | null } = {
  el: null, setter: null,
};
function stopCurrentAudio() {
  if (_currentAudio.el) {
    _currentAudio.el.pause();
    _currentAudio.el.currentTime = 0;
    _currentAudio.el = null;
  }
  if (_currentAudio.setter) {
    _currentAudio.setter(false);
    _currentAudio.setter = null;
  }
}

function TrafficEventCard({ ev }: { ev: TrafficEvent }) {
  const [expanded, setExpanded] = useState(false);
  const [lang, setLang]         = useState<string>("fr");
  const [ttsLoading, setTtsLoading] = useState(false);
  const [playing, setPlaying]       = useState(false);
  const color = SEVERITY_COLOR[ev.severity] ?? C.muted;
  const hintFr = cleanHint(ev.location_hint);
  const direction = ev.direction ? cleanDirection(ev.direction) : "";
  const LIMIT = 160;

  const availableLangs = ["fr", ...Object.keys(ev.translations ?? {})];
  const hint = lang === "fr" ? hintFr : (ev.translations?.[lang] ?? "");
  const isLong = hint.length > LIMIT;

  async function playTTS() {
    if (ttsLoading || !hint) return;
    if (playing) { stopCurrentAudio(); return; }

    stopCurrentAudio();
    setTtsLoading(true);
    try {
      const blob = await synthesizeTTS(hint, lang === "fr" ? "en" : lang);
      const url  = URL.createObjectURL(blob);
      const audio = new Audio(url);
      _currentAudio.el = audio;
      _currentAudio.setter = setPlaying;
      setPlaying(true);
      audio.onended = () => {
        URL.revokeObjectURL(url);
        setPlaying(false);
        if (_currentAudio.el === audio) _currentAudio.el = null;
      };
      await audio.play();
    } catch (e) {
      console.error(e);
      setPlaying(false);
    } finally {
      setTtsLoading(false);
    }
  }

  return (
    <div style={{
      padding: "14px 16px", borderRadius: "14px",
      background: `${color}0a`,
      border: `1px solid ${color}30`,
      display: "flex", flexDirection: "column", gap: "8px",
    }}>
      {/* Ligne 1 : badge(s) type(s) + routes + heure */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
        {/* Affiche tous les types détectés (fusionnés sur la même portion) */}
        {(ev.types && ev.types.length > 0 ? ev.types : [ev.type]).map((t, i) => (
          <span key={`${t}-${i}`} style={{
            fontSize: "11px", fontWeight: 700, letterSpacing: "0.08em",
            padding: "3px 10px", borderRadius: "999px",
            background: `${color}20`, color,
            whiteSpace: "nowrap",
          }}>
            <TrafficTypeIcon type={t} color={color} />
            {" "}{TYPE_LABEL[t] ?? t}
          </span>
        ))}
        {ev.routes.length > 0 && (
          <span style={{
            fontSize: "12px", fontWeight: 700, color: C.fg,
            padding: "2px 8px", borderRadius: "6px",
            background: "rgba(255,255,255,0.06)",
          }}>
            {ev.routes.join(" · ")}
          </span>
        )}
        <span style={{ fontSize: "11px", color: C.muted, marginLeft: "auto", whiteSpace: "nowrap" }}>
          {ev.timestamp}
        </span>
      </div>

      {/* Ligne 2 : direction + délai */}
      {(direction || ev.delay_hint) && (
        <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
          {direction && (
            <span style={{ fontSize: "12px", color: C.muted, display: "flex", alignItems: "center", gap: "4px" }}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 13, height: 13, flexShrink: 0 }}>
                <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/><circle cx="12" cy="10" r="3"/>
              </svg>
              {direction}
            </span>
          )}
          {ev.delay_hint && (
            <span style={{ fontSize: "12px", color, fontWeight: 600, display: "flex", alignItems: "center", gap: "4px" }}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 13, height: 13, flexShrink: 0 }}>
                <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
              </svg>
              {ev.delay_hint}
            </span>
          )}
        </div>
      )}

      {/* Onglets langues + bouton TTS */}
      {availableLangs.length > 1 && (
        <div style={{ display: "flex", gap: "4px", alignItems: "center", flexWrap: "wrap" }}>
          {availableLangs.map(l => (
            <button
              key={l}
              onClick={() => setLang(l)}
              style={{
                fontSize: "11px", fontWeight: 600, letterSpacing: "0.05em",
                padding: "3px 8px", borderRadius: "6px",
                background: lang === l ? `${color}25` : "transparent",
                color:      lang === l ? color : C.muted,
                border: `1px solid ${lang === l ? color : C.border}`,
                cursor: "pointer",
              }}
            >
              {LANG_FLAGS[l] ?? ""} {LANG_LABELS[l] ?? l.toUpperCase()}
            </button>
          ))}
          <button
            onClick={playTTS}
            disabled={ttsLoading || !hint || lang === "fr"}
            title={lang === "fr" ? "TTS disponible en EN/UK/ES" : (playing ? "Arrêter" : "Écouter")}
            style={{
              marginLeft: "auto",
              fontSize: "12px", padding: "3px 10px", borderRadius: "6px",
              background: playing ? `${color}35` : (ttsLoading ? C.border : `${color}15`),
              color, border: `1px solid ${color}40`,
              cursor: (ttsLoading || lang === "fr") ? "not-allowed" : "pointer",
              opacity: lang === "fr" ? 0.4 : 1,
              display: "flex", alignItems: "center", gap: "4px",
            }}
          >
            {playing ? (
              <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: 12, height: 12 }}>
                <rect x="6" y="5" width="4" height="14" rx="1"/>
                <rect x="14" y="5" width="4" height="14" rx="1"/>
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 12, height: 12 }}>
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                <path d="M19.07 4.93a10 10 0 010 14.14M15.54 8.46a5 5 0 010 7.07"/>
              </svg>
            )}
            {ttsLoading ? "…" : (playing ? "Stop" : "Écouter")}
          </button>
        </div>
      )}

      {/* Ligne 3 : transcription complète avec expand */}
      {hint && (
        <div style={{
          borderLeft: `2px solid ${color}40`, paddingLeft: "10px",
        }}>
          <p style={{
            fontSize: "11px", color: C.muted, lineHeight: 1.6, margin: 0,
            fontStyle: "italic",
          }}>
            {expanded || !isLong ? hint : hint.slice(0, LIMIT) + "…"}
          </p>
          {isLong && (
            <button
              onClick={() => setExpanded(v => !v)}
              style={{
                marginTop: "4px", background: "none", border: "none",
                cursor: "pointer", fontSize: "11px", color,
                padding: 0, textDecoration: "underline",
              }}
            >
              {expanded ? "Réduire" : "Lire la suite"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function ZoneColumn({ zone, events }: { zone: string; events: TrafficEvent[] }) {
  const label = zone.charAt(0).toUpperCase() + zone.slice(1);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
        <span style={{ fontSize: "11px", letterSpacing: "0.2em", textTransform: "uppercase", color: C.muted }}>
          Zone {label}
        </span>
        <span style={{
          fontSize: "10px", padding: "1px 7px", borderRadius: "999px",
          background: events.length > 0 ? "rgba(232,112,112,0.12)" : "rgba(126,201,160,0.1)",
          color: events.length > 0 ? C.red : C.green,
        }}>
          {events.length} événement{events.length !== 1 ? "s" : ""}
        </span>
      </div>
      {events.length === 0 ? (
        <div style={{
          padding: "20px", borderRadius: "12px", textAlign: "center",
          border: `1px dashed ${C.border}`, color: C.muted, fontSize: "12px",
        }}>
          Aucun incident détecté
        </div>
      ) : (
        [...events].reverse().map((ev, i) => <TrafficEventCard key={i} ev={ev} />)
      )}
    </div>
  );
}

interface TrafficTabProps {
  snapshot:   TrafficSnapshot;
  connected:  boolean;
  lastUpdate: Date | null;
}
function TrafficTab({ snapshot, connected, lastUpdate }: TrafficTabProps) {
  const [filter, setFilter] = useState<"all" | "urgent">("all");

  const filterFn = (ev: TrafficEvent) =>
    filter === "all" ? true : ev.severity === "high";

  const filteredSnapshot: TrafficSnapshot = {
    nord:  (snapshot.nord  ?? []).filter(filterFn),
    sud:   (snapshot.sud   ?? []).filter(filterFn),
    ouest: (snapshot.ouest ?? []).filter(filterFn),
  };

  const totalShown   = Object.values(filteredSnapshot).flat().length;
  const totalAll     = Object.values(snapshot).flat().length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>

      {/* Status bar */}
      <div style={{
        display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap",
        padding: "10px 16px", borderRadius: "12px",
        background: C.surface, border: `1px solid ${C.border}`,
        fontSize: "12px",
      }}>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{
            width: "8px", height: "8px", borderRadius: "50%", display: "inline-block",
            background: connected ? C.green : C.red,
            animation: connected ? "pulse-dot 2s infinite" : "none",
          }} />
          <span style={{ color: connected ? C.green : C.red }}>
            {connected ? "Flux en direct" : "Reconnexion…"}
          </span>
        </span>
        <span style={{ color: C.muted }}>·</span>
        <span style={{ color: C.muted }}>
          {totalShown} sur {totalAll} évènement{totalAll !== 1 ? "s" : ""}
        </span>
        {lastUpdate && (
          <>
            <span style={{ color: C.muted }}>·</span>
            <span style={{ color: C.muted }}>
              Mis à jour {lastUpdate.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
            </span>
          </>
        )}

        {/* Toggle filtre */}
        <div style={{
          marginLeft: "auto", display: "flex", gap: "4px",
          padding: "3px", borderRadius: "999px", background: "var(--background)",
          border: `1px solid ${C.border}`,
        }}>
          {([
            { key: "all",    label: "Tous les flashs" },
            { key: "urgent", label: "Urgences uniquement" },
          ] as const).map(opt => (
            <button key={opt.key} onClick={() => setFilter(opt.key)} style={{
              padding: "5px 12px", borderRadius: "999px", fontSize: "11px",
              cursor: "pointer", border: "none", fontWeight: 500,
              background: filter === opt.key ? "rgba(201,169,110,0.15)" : "transparent",
              color: filter === opt.key ? C.accent : C.muted,
              transition: "all 0.15s",
            }}>
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* 3 colonnes zones */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px" }}>
        <ZoneColumn zone="nord"  events={filteredSnapshot.nord} />
        <ZoneColumn zone="sud"   events={filteredSnapshot.sud} />
        <ZoneColumn zone="ouest" events={filteredSnapshot.ouest} />
      </div>

    </div>
  );
}

// ── Tab list item ─────────────────────────────────────────────────────────────
const TAB_ITEMS = [
  { value: "overview",     label: "Vue générale" },
  { value: "traces",       label: "Traces & Modèles" },
  { value: "traffic",      label: "Trafic Live" },
  { value: "experiments",  label: "Expériences" },
  { value: "infra",        label: "Infrastructure" },
  { value: "pipelines",    label: "Pipelines" },
  { value: "users",        label: "Utilisateurs" },
];

// ── Page ──────────────────────────────────────────────────────────────────────
export default function AdminPage() {
  const router = useRouter();

  const [currentUserId, setCurrentUserId] = useState<number | null>(null);
  const [stats,    setStats]    = useState<AdminStats | null>(null);
  const [users,    setUsers]    = useState<AdminUser[]>([]);
  const [langfuse, setLangfuse] = useState<LangfuseMetrics | null>(null);
  const [loadingLf, setLoadingLf] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");

  // Traffic live state — monté dans AdminPage pour survivre aux switch d'onglets
  const [trafficSnapshot, setTrafficSnapshot] = useState<TrafficSnapshot>({ nord: [], sud: [], ouest: [] });
  const [trafficConnected, setTrafficConnected] = useState(false);
  const [trafficLastUpdate, setTrafficLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    if (!currentUserId) return;
    getTrafficEvents().then(setTrafficSnapshot).catch(() => {});
    const es = openTrafficStream((data) => {
      setTrafficLastUpdate(new Date());
      if ("snapshot" in data) {
        setTrafficSnapshot(data.snapshot as TrafficSnapshot);
      } else if ("zone" in data && "events" in data) {
        const { zone, events } = data as { zone: string; events: TrafficEvent[] };
        setTrafficSnapshot(prev => ({ ...prev, [zone]: events }));
      }
    });
    es.onopen  = () => setTrafficConnected(true);
    es.onerror = () => setTrafficConnected(false);
    return () => es.close();
  }, [currentUserId]);

  // Auth check + load initial data
  useEffect(() => {
    getMe()
      .then(u => {
        if (!u || !u.is_admin) { router.push("/"); return; }
        setCurrentUserId(u.id);
        loadStats();
        loadUsers();
      })
      .catch(() => router.push("/login"));
  }, []);

  const loadStats = useCallback(async () => {
    try { setStats(await getAdminStats()); } catch {}
  }, []);

  const loadUsers = useCallback(async () => {
    try { setUsers(await getAdminUsers()); } catch {}
  }, []);

  const loadLangfuse = useCallback(async () => {
    if (langfuse) return;
    setLoadingLf(true);
    try { setLangfuse(await getLangfuseMetrics()); } catch {}
    finally { setLoadingLf(false); }
  }, [langfuse]);

  // Lazy-load Langfuse when its tab is opened
  useEffect(() => {
    if (activeTab === "traces" && !langfuse) loadLangfuse();
    if (activeTab === "overview" && !langfuse) loadLangfuse();
  }, [activeTab]);

  if (!currentUserId) return null;

  return (
    <main style={{
      minHeight: "100vh", background: C.bg,
      padding: "32px 24px 80px",
      display: "flex", flexDirection: "column", alignItems: "center",
    }}>
      <div style={{ width: "100%", maxWidth: "900px" }}>

        {/* Tabs */}
        <Tabs.Root value={activeTab} onValueChange={setActiveTab}>

          {/* Sticky header + tab list */}
          <div style={{
            position: "sticky", top: 0, zIndex: 10,
            background: C.bg,
            paddingTop: "32px",
            marginTop: "-32px",
            paddingBottom: "4px",
          }}>
            {/* Header */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "20px" }}>
              <div>
                <div style={{
                  display: "inline-block", fontSize: "10px", letterSpacing: "0.35em",
                  textTransform: "uppercase", marginBottom: "10px",
                  padding: "4px 12px", borderRadius: "999px",
                  background: "rgba(201,169,110,0.08)", color: C.accent,
                }}>
                  Administration
                </div>
                <h1 className="font-serif" style={{ fontSize: "clamp(22px, 4vw, 32px)", color: C.fg, lineHeight: 1.2 }}>
                  Dashboard <em style={{ color: C.accent }}>MLOps</em>
                </h1>
              </div>
              <button onClick={() => router.push("/")} style={{
                padding: "7px 16px", borderRadius: "999px", fontSize: "12px",
                cursor: "pointer", background: C.surface, border: `1px solid ${C.border}`, color: C.muted,
              }}>
                ← Retour
              </button>
            </div>

            <Tabs.List style={{
              display: "flex", gap: "4px", flexWrap: "wrap",
              padding: "4px", borderRadius: "14px",
              background: C.surface, border: `1px solid ${C.border}`,
              marginBottom: "24px",
            }}>
              {TAB_ITEMS.map(t => (
                <Tabs.Trigger key={t.value} value={t.value} style={{
                  padding: "7px 16px", borderRadius: "10px", fontSize: "12px",
                  cursor: "pointer", border: "none", fontWeight: 500,
                  background: activeTab === t.value ? "rgba(201,169,110,0.12)" : "none",
                  color: activeTab === t.value ? C.accent : C.muted,
                  transition: "all 0.15s",
                }}>
                  {t.label}
                </Tabs.Trigger>
              ))}
            </Tabs.List>
          </div>

          <Tabs.Content value="overview">
            <OverviewTab stats={stats} langfuse={langfuse} />
          </Tabs.Content>

          <Tabs.Content value="traces">
            {loadingLf ? <Loader /> : <TracesTab langfuse={langfuse} />}
          </Tabs.Content>

          <Tabs.Content value="traffic">
            <TrafficTab
              snapshot={trafficSnapshot}
              connected={trafficConnected}
              lastUpdate={trafficLastUpdate}
            />
          </Tabs.Content>

          <Tabs.Content value="experiments">
            <ExperimentsTab />
          </Tabs.Content>

          <Tabs.Content value="infra">
            <InfraTab />
          </Tabs.Content>

          <Tabs.Content value="pipelines">
            <PipelinesTab />
          </Tabs.Content>

          <Tabs.Content value="users">
            <UsersTab users={users} currentUserId={currentUserId} onRefresh={loadUsers} />
          </Tabs.Content>
        </Tabs.Root>
      </div>

      <footer style={{
        position: "fixed", bottom: 0, left: 0, right: 0, padding: "12px 24px",
        textAlign: "center", borderTop: `1px solid ${C.border}`, background: C.bg, zIndex: 10,
      }}>
        <p style={{ fontSize: "11px", letterSpacing: "0.12em", color: C.muted, opacity: 0.4 }}>
          © {new Date().getFullYear()} traduction-audio.fr · Admin MLOps Dashboard
        </p>
      </footer>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.3; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1.2); }
        }
        @keyframes pulse-dot {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </main>
  );
}
