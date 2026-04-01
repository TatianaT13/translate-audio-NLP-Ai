"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import * as Tabs from "@radix-ui/react-tabs";
import { getMe } from "@/lib/auth";
import {
  getAdminStats, getAdminUsers, getLangfuseMetrics, updateAdminUser, deleteAdminUser,
  getServicesHealth,
  type AdminUser, type AdminStats, type LangfuseMetrics, type LangfuseModelStat,
  type ServiceHealth,
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
function TracesTab({ langfuse }: { langfuse: LangfuseMetrics | null }) {
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

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>

      {/* Latency bar chart */}
      <Card title="Latences moyennes par service">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "20px", alignItems: "end" }}>
          {latencyData.map(d => (
            <div key={d.label} style={{ textAlign: "center" }}>
              <div style={{
                height: "80px", display: "flex", alignItems: "flex-end", justifyContent: "center",
              }}>
                <div style={{
                  width: "48px",
                  height: `${Math.round((d.value / Math.max(langfuse.avg_total_ms, 1)) * 80)}px`,
                  borderRadius: "6px 6px 0 0",
                  background: d.label === "STT" ? C.stt : d.label === "LLM" ? C.llm : C.muted,
                  opacity: 0.85, minHeight: "4px",
                  transition: "height 0.6s ease",
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
        <Card title="Distribution score BLEU">
          {langfuse.bleu_scores.length > 0 ? (
            <>
              <Histogram values={langfuse.bleu_scores} bins={8} color={C.tts} height={60} />
              <p style={{ fontSize: "11px", color: C.muted, marginTop: "8px" }}>
                Moy. {langfuse.avg_bleu.toFixed(3)} · {langfuse.bleu_scores.length} traces
              </p>
            </>
          ) : (
            <p style={{ fontSize: "12px", color: C.muted }}>Aucun score BLEU enregistré</p>
          )}
        </Card>
        <Card title="Distribution score METEOR">
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

      {/* Model comparison table */}
      {langfuse.model_stats.length > 0 && (
        <Card title="Comparaison des modèles">
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
              <thead>
                <tr>
                  {["Whisper", "LLM", "Prompt", "Runs", "STT moy.", "LLM moy.", "Total moy.", "BLEU moy.", "METEOR moy.", "WER moy."].map(h => (
                    <th key={h} style={{
                      padding: "8px 12px", textAlign: "left", fontWeight: 600,
                      letterSpacing: "0.1em", color: C.muted, fontSize: "10px",
                      borderBottom: `1px solid ${C.border}`,
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {langfuse.model_stats.map((m, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${C.border}` }}>
                    <td style={{ padding: "10px 12px", color: C.stt }}>{m.whisper}</td>
                    <td style={{ padding: "10px 12px", color: C.llm, maxWidth: "160px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.llm}</td>
                    <td style={{ padding: "10px 12px", color: C.muted }}>{m.prompt_version}</td>
                    <td style={{ padding: "10px 12px", color: C.fg, fontWeight: 600 }}>{m.count}</td>
                    <td style={{ padding: "10px 12px", color: C.stt, fontVariantNumeric: "tabular-nums" }}>{ms(m.avg_stt_ms)}</td>
                    <td style={{ padding: "10px 12px", color: C.llm, fontVariantNumeric: "tabular-nums" }}>{ms(m.avg_llm_ms)}</td>
                    <td style={{ padding: "10px 12px", color: C.muted, fontVariantNumeric: "tabular-nums" }}>{ms(m.avg_total_ms)}</td>
                    <td style={{ padding: "10px 12px", color: C.tts }}>
                      {m.avg_bleu != null ? m.avg_bleu.toFixed(3) : <span style={{ opacity: 0.4 }}>—</span>}
                    </td>
                    <td style={{ padding: "10px 12px", color: C.green }}>
                      {m.avg_meteor != null ? m.avg_meteor.toFixed(4) : <span style={{ opacity: 0.4 }}>—</span>}
                    </td>
                    <td style={{ padding: "10px 12px", color: C.red }}>
                      {m.avg_wer != null ? m.avg_wer.toFixed(4) : <span style={{ opacity: 0.4 }}>—</span>}
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

// ── Experiments tab (MLflow stub) ─────────────────────────────────────────────
function ExperimentsTab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div style={{
        padding: "14px 18px", borderRadius: "12px", fontSize: "12px",
        background: "rgba(201,169,110,0.06)", border: "1px solid var(--accent-dim)", color: C.accent,
        letterSpacing: "0.05em",
      }}>
        Phase suivante du roadmap MLOps — intégration MLflow en cours
      </div>
      <ComingSoon
        title="MLflow — Model Registry"
        description="Comparaison d'expériences, versioning des modèles, métriques d'entraînement et de déploiement. Lance MLflow avec docker compose --profile mlflow up."
        icon={
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>
          </svg>
        }
      />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
        {[
          { label: "Expériences", value: "—", desc: "MLflow requis" },
          { label: "Runs enregistrés", value: "84", desc: "Importés via Langfuse" },
          { label: "Modèles en registry", value: "—", desc: "MLflow requis" },
          { label: "Meilleur BLEU", value: "—", desc: "Via Langfuse scores" },
        ].map(c => (
          <StatCard key={c.label} label={c.label} value={c.value} sub={c.desc} />
        ))}
      </div>
    </div>
  );
}

// ── Infrastructure tab ────────────────────────────────────────────────────────
function InfraTab() {
  const [services, setServices] = useState<ServiceHealth[]>([]);
  const [loading,  setLoading]  = useState(true);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getServicesHealth();
      setServices(data.services);
      setLastCheck(new Date());
    } catch {
      // gateway down or not admin
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, []);

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

      <ComingSoon
        title="Grafana — Métriques système"
        description="CPU, mémoire, requêtes/sec par service Docker. Dashboards Prometheus + Grafana intégrés au docker-compose."
        icon={
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8m-4-4v4"/>
            <path d="m7 10 3 3 3-3 3 3"/>
          </svg>
        }
      />
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

// ── Tab list item ─────────────────────────────────────────────────────────────
const TAB_ITEMS = [
  { value: "overview",     label: "Vue générale" },
  { value: "traces",       label: "Traces & Modèles" },
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

        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "32px" }}>
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

        {/* Tabs */}
        <Tabs.Root value={activeTab} onValueChange={setActiveTab}>
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

          <Tabs.Content value="overview">
            <OverviewTab stats={stats} langfuse={langfuse} />
          </Tabs.Content>

          <Tabs.Content value="traces">
            {loadingLf ? <Loader /> : <TracesTab langfuse={langfuse} />}
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
