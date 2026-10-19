"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AppHeader } from "@/components/AppHeader";
import { FooterMinimal } from "@/components/Footer";
import {
  getCredits, getTransactions, describeTransaction,
  type CreditsState, type CreditTransaction,
} from "@/lib/credits";

export default function ConsommationPage() {
  const [credits, setCredits] = useState<CreditsState | null>(null);
  const [rows, setRows]       = useState<CreditTransaction[] | null>(null);

  useEffect(() => {
    getCredits().then(setCredits).catch(() => {});
    getTransactions().then(setRows).catch(() => setRows([]));
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
                {credits.period_resets_at && credits.tier !== "free" &&
                  ` · prochaine recharge le ${new Date(credits.period_resets_at).toLocaleDateString("fr-FR")}`}
              </div>
            </div>
            <Link href="/tarifs" style={{ fontSize: "13px", color: "var(--accent)" }}>
              {credits.tier === "free" ? "Passer à Pro" : "Voir les tarifs"}
            </Link>
          </div>
        )}

        <h2 style={{ fontSize: "14px", fontWeight: 600, color: "var(--foreground)", margin: "0 0 12px" }}>Historique</h2>
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
          Les réunions sont décomptées à la durée réelle : 0,5 crédit par minute, débité par crédit entier.
        </p>
      </main>
      <FooterMinimal />
    </>
  );
}
