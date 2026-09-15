"use client";

import Link from "next/link";

// ── Footer minimaliste ────────────────────────────────────────────────────────
// Utilise sur les pages internes (translate, live, meeting) — pied de page
// discret, ne prend qu'une ligne. Annee auto-updatee.
export function FooterMinimal() {
  return (
    <footer style={{
      padding: "16px 32px",
      borderTop: "1px solid var(--border)",
      textAlign: "center",
      background: "var(--background)",
    }}>
      <p style={{
        fontSize: "11px", letterSpacing: "0.12em",
        color: "var(--muted)", opacity: 0.5,
        margin: 0,
      }}>
        © {new Date().getFullYear()} traduction-audio.fr — Tous droits réservés
      </p>
    </footer>
  );
}

// ── Footer pro complet ────────────────────────────────────────────────────────
// Utilise sur la landing publique et pages legales. Colonnes Produit /
// Entreprise / Legal. Zone basse : copyright + mentions administratives.
export function FooterFull() {
  const columns: { title: string; links: { label: string; href: string; external?: boolean }[] }[] = [
    {
      title: "Produit",
      links: [
        { label: "Traduction",       href: "/translate" },
        { label: "Traduction live",  href: "/live" },
        { label: "Compte-rendu",     href: "/meeting" },
        { label: "Tarifs",           href: "/tarifs" },
      ],
    },
    {
      title: "Entreprise",
      links: [
        { label: "À propos", href: "/a-propos" },
        { label: "Contact",  href: "/contact" },
      ],
    },
    {
      title: "Légal",
      links: [
        { label: "Mentions légales",         href: "/mentions-legales" },
        { label: "Conditions d'utilisation", href: "/cgu" },
        { label: "Conditions de vente",      href: "/cgv" },
        { label: "Politique de confidentialité", href: "/confidentialite" },
        { label: "Gestion des cookies",      href: "/cookies" },
      ],
    },
  ];

  return (
    <footer style={{
      borderTop: "1px solid var(--border)",
      background: "var(--background)",
      padding: "48px 32px 24px",
    }}>
      <div style={{ maxWidth: "1200px", margin: "0 auto" }}>

        {/* Top : Brand + colonnes */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "minmax(220px, 1fr) repeat(3, minmax(160px, 1fr))",
          gap: "40px",
          marginBottom: "40px",
        }}>
          {/* Brand + tagline */}
          <div>
            <Link href="/" style={{
              fontFamily: "var(--font-display), serif",
              fontSize: "18px", color: "var(--foreground)",
              textDecoration: "none", letterSpacing: "-0.01em",
            }}>
              traduction-audio<span style={{ color: "var(--accent)" }}>.</span>
            </Link>
            <p style={{
              fontSize: "12px", color: "var(--muted)",
              marginTop: "12px", lineHeight: 1.55,
              maxWidth: "260px",
            }}>
              Traduction audio à la demande, dans un langage clair, pour comprendre et se faire comprendre.
            </p>
          </div>

          {/* 3 colonnes de liens */}
          {columns.map((col) => (
            <div key={col.title}>
              <h3 style={{
                fontSize: "10px", letterSpacing: "0.22em",
                textTransform: "uppercase", color: "var(--muted)",
                fontWeight: 500, margin: "0 0 14px 0",
              }}>
                {col.title}
              </h3>
              <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: "8px" }}>
                {col.links.map((link) => (
                  <li key={link.label}>
                    <Link
                      href={link.href}
                      style={{
                        fontSize: "13px",
                        color: "var(--foreground)",
                        textDecoration: "none",
                        transition: "color 0.15s",
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.color = "var(--accent)"}
                      onMouseLeave={(e) => e.currentTarget.style.color = "var(--foreground)"}
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Separator */}
        <div style={{ height: "1px", background: "var(--border)", opacity: 0.6, marginBottom: "20px" }} />

        {/* Bottom : copyright + mentions */}
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "16px",
          flexWrap: "wrap",
        }}>
          <p style={{
            fontSize: "11px", letterSpacing: "0.08em",
            color: "var(--muted)", opacity: 0.7,
            margin: 0,
          }}>
            © {new Date().getFullYear()} traduction-audio.fr — Tous droits réservés
          </p>
          <p style={{
            fontSize: "11px", letterSpacing: "0.08em",
            color: "var(--muted)", opacity: 0.55,
            margin: 0,
          }}>
            Made in France
          </p>
        </div>
      </div>

      <style>{`
        @media (max-width: 720px) {
          footer > div > div:first-child {
            grid-template-columns: 1fr !important;
            gap: 32px !important;
          }
        }
      `}</style>
    </footer>
  );
}
