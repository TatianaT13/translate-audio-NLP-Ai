"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AppHeader } from "@/components/AppHeader";
import { FooterMinimal } from "@/components/Footer";
import {
  getCredits, getTransactions, describeTransaction, getDailyUsage, formatDuration, SERVICE_LABEL,
  getSubscription, cancelSubscription, resumeSubscription,
  type CreditsState, type CreditTransaction, type DailyUsage, type SubscriptionState,
} from "@/lib/credits";

export default function ConsommationPage() {
  const [credits, setCredits] = useState<CreditsState | null>(null);
  const [rows, setRows]       = useState<CreditTransaction[] | null>(null);
  const [usage, setUsage]     = useState<DailyUsage[] | null>(null);
  const [subscription, setSubscription] = useState<SubscriptionState | null>(null);

  useEffect(() => {
    getCredits().then(setCredits).catch(() => {});
    getTransactions().then(setRows).catch(() => setRows([]));
    getDailyUsage().then(setUsage).catch(() => setUsage([]));
    getSubscription().then(setSubscription).catch(() => {});
  }, []);

  const fmtDate = (iso: string | null) =>
    iso ? new Date(iso).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" }) : "";

  return (
    <>
      <AppHeader />
      <main style={{ maxWidth: "720px", margin: "0 auto", padding: "40px 24px 64px" }}>
        <h1 className="font-serif" style={{ fontSize: "clamp(26px, 4vw, 34px)", color: "var(--foreground)", margin: "0 0 24px" }}>
          Ma consommation
        </h1>

        {credits && (
          <div style={{
            display: "flex", flexWrap: "wrap", gap: "24px", alignItems: "center", justifyContent: "space-between",
            padding: "20px 24px", borderRadius: "16px", marginBottom: "32px",
            background: "var(--surface)", border: "1px solid var(--border)",
          }}>
            <div>
              <div style={{ fontSize: "32px", color: "var(--foreground)", fontVariantNumeric: "tabular-nums" }}>
                {credits.processing.balance} <span style={{ fontSize: "14px", color: "var(--muted)" }}>crédits</span>
              </div>
              <div style={{ fontSize: "12px", color: "var(--muted)", marginTop: "4px" }}>
                {credits.tier === "free"
                  ? "Offre Découverte"
                  : `Offre Pro · ${credits.processing.monthly} crédits par mois`}
                {credits.period_resets_at && credits.tier !== "free" && !subscription?.cancel_at_period_end &&
                  ` · prochaine recharge le ${new Date(credits.period_resets_at).toLocaleDateString("fr-FR")}`}
              </div>
            </div>
            <Link href="/tarifs" style={{ fontSize: "13px", color: "var(--accent)" }}>
              {credits.tier === "free" ? "Passer à Pro" : "Voir les tarifs"}
            </Link>
          </div>
        )}

        {subscription && subscription.tier !== "free" && (
          <SubscriptionPanel subscription={subscription} onChange={setSubscription} />
        )}

        <h2 style={{ fontSize: "14px", fontWeight: 600, color: "var(--foreground)", margin: "0 0 12px" }}>
          Utilisation par jour <span style={{ fontWeight: 400, color: "var(--muted)" }}>· 90 derniers jours</span>
        </h2>
        {usage !== null && usage.length === 0 && (
          <p style={{ color: "var(--muted)", fontSize: "13px", marginBottom: "32px" }}>Aucune utilisation pour l&apos;instant.</p>
        )}
        {usage !== null && usage.length > 0 && (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px", marginBottom: "40px" }}>
            <thead>
              <tr style={{ color: "var(--muted)", textAlign: "left" }}>
                <th style={{ padding: "8px 0", fontWeight: 500 }}>Date</th>
                <th style={{ padding: "8px 0", fontWeight: 500 }}>Service</th>
                <th style={{ padding: "8px 0", fontWeight: 500, textAlign: "right" }}>Durée</th>
                <th style={{ padding: "8px 0", fontWeight: 500, textAlign: "right" }}>Crédits</th>
              </tr>
            </thead>
            <tbody>
              {usage.map((u) => (
                <tr key={`${u.date}-${u.service}`} style={{ borderTop: "1px solid var(--border)" }}>
                  <td style={{ padding: "10px 12px 10px 0", color: "var(--muted)", whiteSpace: "nowrap" }}>
                    {new Date(u.date + "T12:00:00").toLocaleDateString("fr-FR")}
                  </td>
                  <td style={{ padding: "10px 12px 10px 0", color: "var(--foreground)" }}>{SERVICE_LABEL[u.service]}</td>
                  <td style={{ padding: "10px 0", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{formatDuration(u)}</td>
                  <td style={{ padding: "10px 0", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{u.credits}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <h2 style={{ fontSize: "14px", fontWeight: 600, color: "var(--foreground)", margin: "0 0 12px" }}>Mouvements de crédits</h2>
        {rows === null && <p style={{ color: "var(--muted)", fontSize: "13px" }}>Chargement…</p>}
        {rows !== null && rows.length === 0 && (
          <p style={{ color: "var(--muted)", fontSize: "13px" }}>Aucun mouvement pour l&apos;instant.</p>
        )}
        {rows !== null && rows.length > 0 && (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
            <thead>
              <tr style={{ color: "var(--muted)", textAlign: "left" }}>
                <th style={{ padding: "8px 0", fontWeight: 500 }}>Date</th>
                <th style={{ padding: "8px 0", fontWeight: 500 }}>Opération</th>
                <th style={{ padding: "8px 0", fontWeight: 500, textAlign: "right" }}>Crédits</th>
                <th style={{ padding: "8px 0", fontWeight: 500, textAlign: "right" }}>Solde</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((t, i) => (
                <tr key={i} style={{ borderTop: "1px solid var(--border)" }}>
                  <td style={{ padding: "10px 12px 10px 0", color: "var(--muted)", whiteSpace: "nowrap" }}>{fmtDate(t.date)}</td>
                  <td style={{ padding: "10px 12px 10px 0", color: "var(--foreground)" }}>{describeTransaction(t)}</td>
                  <td style={{
                    padding: "10px 0", textAlign: "right", fontVariantNumeric: "tabular-nums",
                    color: t.amount >= 0 ? "#7ec9a0" : "var(--foreground)",
                  }}>{t.amount > 0 ? `+${t.amount}` : t.amount}</td>
                  <td style={{ padding: "10px 0", textAlign: "right", color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>{t.balance_after}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p style={{ fontSize: "12px", color: "var(--muted)", marginTop: "20px", lineHeight: 1.6 }}>
          Les réunions sont décomptées à la durée réelle (0,5 crédit par minute) et débitées par crédit
          entier, soit 1 crédit toutes les 2 minutes de réunion cumulées.
        </p>
      </main>
      <FooterMinimal />
    </>
  );
}

// ── Abonnement : résiliation en ligne (art. L. 215-1-1 du Code de la consommation) ──

function SubscriptionPanel({ subscription, onChange }: {
  subscription: SubscriptionState;
  onChange: (s: SubscriptionState) => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy]             = useState(false);
  const [error, setError]           = useState<string | null>(null);

  const end = subscription.current_period_end
    ? new Date(subscription.current_period_end).toLocaleDateString("fr-FR")
    : null;

  const run = async (action: () => Promise<SubscriptionState>) => {
    setBusy(true);
    setError(null);
    try {
      onChange(await action());
      setConfirming(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Opération impossible");
    } finally {
      setBusy(false);
    }
  };

  const box: React.CSSProperties = {
    padding: "18px 24px", borderRadius: "16px", marginBottom: "32px",
    background: "var(--surface)", border: "1px solid var(--border)", fontSize: "13px", lineHeight: 1.6,
  };
  const action: React.CSSProperties = {
    background: "none", border: "none", padding: 0, font: "inherit", fontSize: "13px",
    cursor: busy ? "wait" : "pointer", textDecoration: "underline",
  };

  return (
    <div style={box}>
      <div style={{ fontSize: "14px", fontWeight: 600, color: "var(--foreground)", marginBottom: "6px" }}>
        Mon abonnement
      </div>

      {subscription.cancel_at_period_end ? (
        <>
          <p style={{ color: "var(--foreground)", margin: "0 0 10px" }}>
            Résiliation enregistrée. Votre abonnement Pro reste actif{end ? ` jusqu'au ${end}` : " jusqu'à la fin de la période en cours"},
            puis s&apos;arrête : aucun nouveau prélèvement ne sera effectué. Vos crédits restants resteront utilisables.
          </p>
          <button onClick={() => run(resumeSubscription)} disabled={busy} style={{ ...action, color: "var(--accent)" }}>
            Reprendre mon abonnement
          </button>
        </>
      ) : confirming ? (
        <>
          <p style={{ color: "var(--foreground)", margin: "0 0 4px" }}>Vous allez résilier votre abonnement Pro.</p>
          <ul style={{ color: "var(--muted)", margin: "0 0 14px", paddingLeft: "18px" }}>
            <li>Il reste actif{end ? ` jusqu'au ${end}` : " jusqu'à la fin de la période déjà payée"}.</li>
            <li>Aucun nouveau prélèvement ne sera effectué ensuite.</li>
            <li>La période en cours n&apos;est pas remboursée ; vos crédits restants resteront utilisables.</li>
          </ul>
          <div style={{ display: "flex", gap: "20px", flexWrap: "wrap" }}>
            <button onClick={() => run(cancelSubscription)} disabled={busy} style={{ ...action, color: "#e87070" }}>
              {busy ? "Résiliation…" : "Confirmer la résiliation"}
            </button>
            <button onClick={() => setConfirming(false)} disabled={busy} style={{ ...action, color: "var(--muted)" }}>
              Garder mon abonnement
            </button>
          </div>
        </>
      ) : (
        <>
          <p style={{ color: "var(--muted)", margin: "0 0 10px" }}>
            Offre Pro{end ? ` · prochain renouvellement le ${end}` : ""}. Sans engagement.
          </p>
          <button onClick={() => setConfirming(true)} style={{ ...action, color: "var(--muted)" }}>
            Résilier mon abonnement
          </button>
        </>
      )}

      {error && <p style={{ color: "#e87070", margin: "10px 0 0" }}>{error}</p>}
    </div>
  );
}
