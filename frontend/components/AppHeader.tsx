"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import {
  getMe, logout, changePassword, deleteAccount,
  refreshAccessToken, checkPasswordStrength,
} from "@/lib/auth";
import type { User } from "@/lib/auth";
import {
  TranslateIcon, MicIcon, DocumentIcon,
  HomeIcon, AdminIcon, LogoutIcon, KeyIcon, TrashIcon,
  ChevronDownIcon,
} from "@/components/icons";

// ── Types ────────────────────────────────────────────────────────────────────

type Variant = "landing" | "feature";
type Current = "translate" | "live" | "meeting" | "admin" | null;

interface AppHeaderProps {
  /** "landing" affiche uniquement logo + user area (les cards sont sous). */
  variant?: Variant;
  /** Highlight de l'onglet actif sur les pages feature. */
  current?: Current;
}

const NAV_ITEMS: { key: Exclude<Current, null>; label: string; href: string; Icon: React.FC<{ size?: number }> }[] = [
  { key: "translate", label: "Traduction",       href: "/translate", Icon: TranslateIcon },
  { key: "live",      label: "Live",             href: "/live",      Icon: MicIcon },
  { key: "meeting",   label: "Réunion",          href: "/meeting",   Icon: DocumentIcon },
];

// ── Component ────────────────────────────────────────────────────────────────

export function AppHeader({ variant = "feature", current = null }: AppHeaderProps) {
  const router   = useRouter();
  const pathname = usePathname();

  const [user, setUser]           = useState<User | null>(null);
  const [checking, setChecking]   = useState(true);
  const [menuOpen, setMenuOpen]   = useState(false);
  const [showPwd, setShowPwd]     = useState(false);
  const [showDel, setShowDel]     = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Detecter le mode "current" automatiquement si non fourni
  const activeKey: Current = current ?? detectCurrent(pathname);

  // Fetch user + refresh preventif JWT
  useEffect(() => {
    getMe().then(setUser).catch(() => setUser(null)).finally(() => setChecking(false));
    const id = setInterval(() => { refreshAccessToken().catch(() => {}); }, 10 * 60 * 1000);
    return () => clearInterval(id);
  }, []);

  // Fermer le dropdown au clic exterieur
  useEffect(() => {
    if (!menuOpen) return;
    const onClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [menuOpen]);

  const handleLogout = async () => {
    setMenuOpen(false);
    await logout().catch(() => {});
    router.push("/");
  };

  const handleChangeSuccess = () => {
    setShowPwd(false);
    router.push("/login");
  };

  return (
    <>
      {showPwd && <ChangePasswordModal onClose={() => setShowPwd(false)} onSuccess={handleChangeSuccess} />}
      {showDel && <DeleteAccountModal onClose={() => setShowDel(false)} onDeleted={() => router.push("/")} />}

      <header style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "16px 32px",
        borderBottom: "1px solid var(--border)",
        background: "var(--background)",
        position: "sticky", top: 0, zIndex: 30,
        backdropFilter: "blur(8px)",
      }}>
        {/* Brand */}
        <Link href="/" style={{
          display: "flex", alignItems: "center", gap: "10px",
          fontFamily: "var(--font-display), serif",
          fontSize: "17px", color: "var(--foreground)",
          textDecoration: "none", letterSpacing: "-0.01em",
        }}>
          <HomeIcon size={18} />
          <span>traduction-audio<span style={{ color: "var(--accent)" }}>.</span></span>
        </Link>

        {/* Nav centrale — feature only */}
        {variant === "feature" && (
          <nav style={{ display: "flex", gap: "4px" }}>
            {NAV_ITEMS.map(({ key, label, href, Icon }) => {
              const active = activeKey === key;
              return (
                <Link key={key} href={href} style={{
                  display: "inline-flex", alignItems: "center", gap: "8px",
                  padding: "8px 14px", borderRadius: "10px",
                  fontSize: "13px",
                  color: active ? "var(--accent)" : "var(--muted)",
                  background: active ? "rgba(201,169,110,0.10)" : "transparent",
                  border: `1px solid ${active ? "var(--accent-dim)" : "transparent"}`,
                  textDecoration: "none",
                  transition: "all 0.15s",
                }}>
                  <Icon size={16} />
                  <span>{label}</span>
                </Link>
              );
            })}
          </nav>
        )}

        {/* User area */}
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          {checking ? (
            <div style={{ width: "120px", height: "32px" }} />
          ) : user ? (
            <div ref={menuRef} style={{ position: "relative" }}>
              <button
                onClick={() => setMenuOpen(o => !o)}
                aria-expanded={menuOpen}
                aria-label="Menu utilisateur"
                style={{
                  display: "flex", alignItems: "center", gap: "8px",
                  padding: "6px 12px 6px 6px", borderRadius: "999px",
                  cursor: "pointer",
                  background: menuOpen ? "rgba(201,169,110,0.1)" : "var(--surface)",
                  border: "1px solid var(--border)",
                  color: "var(--muted)", fontSize: "12px",
                  transition: "all 0.15s",
                }}
              >
                <div style={{
                  width: "24px", height: "24px", borderRadius: "50%",
                  background: "rgba(201,169,110,0.15)",
                  border: "1px solid var(--accent-dim)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: "10px", color: "var(--accent)", fontWeight: 700,
                  textTransform: "uppercase",
                }}>
                  {user.email[0]}
                </div>
                <span style={{ maxWidth: "160px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {user.email}
                </span>
                <ChevronDownIcon
                  size={12}
                  style={{ transition: "transform 0.2s", transform: menuOpen ? "rotate(180deg)" : "none", flexShrink: 0 }}
                />
              </button>

              {menuOpen && (
                <DropdownMenu
                  user={user}
                  onNavigate={(href) => { setMenuOpen(false); router.push(href); }}
                  onChangePwd={() => { setMenuOpen(false); setShowPwd(true); }}
                  onDelete={() => { setMenuOpen(false); setShowDel(true); }}
                  onLogout={handleLogout}
                />
              )}
            </div>
          ) : (
            <>
              <Link href="/login" style={{
                fontSize: "13px", color: "var(--muted)", textDecoration: "none",
              }}>Se connecter</Link>
              <Link href="/register" style={{
                fontSize: "13px", color: "var(--accent)", textDecoration: "none",
                padding: "7px 14px", borderRadius: "8px",
                border: "1px solid var(--accent-dim)",
                background: "rgba(201,169,110,0.06)",
              }}>Créer un compte</Link>
            </>
          )}
        </div>
      </header>
    </>
  );
}

// ── Dropdown menu ────────────────────────────────────────────────────────────

interface DropdownMenuProps {
  user:          User;
  onNavigate:    (href: string) => void;
  onChangePwd:   () => void;
  onDelete:      () => void;
  onLogout:      () => void;
}

function DropdownMenu({ user, onNavigate, onChangePwd, onDelete, onLogout }: DropdownMenuProps) {
  const items: (
    | { type: "link";      label: string; href: string; Icon: React.FC<{ size?: number }>; danger?: boolean }
    | { type: "action";    label: string; onClick: () => void; Icon: React.FC<{ size?: number }>; danger?: boolean }
    | { type: "separator" }
  )[] = [
    { type: "link",   label: "Accueil",                href: "/",          Icon: HomeIcon },
    { type: "separator" },
    ...(user.is_admin ? [{ type: "link" as const, label: "Admin",       href: "/admin",     Icon: AdminIcon }] : []),
    // Support de soutenance : masque du menu utilisateur (garde sur disque et
    // sur git dans frontend/public/soutenance-pitch.html pour reference).
    // { type: "link",   label: "Support de soutenance", href: "/soutenance-pitch.html", Icon: ExternalLinkIcon },
    { type: "separator" },
    { type: "action", label: "Changer le mot de passe", onClick: onChangePwd, Icon: KeyIcon },
    { type: "action", label: "Se déconnecter",          onClick: onLogout,     Icon: LogoutIcon },
    { type: "separator" },
    { type: "action", label: "Supprimer le compte",     onClick: onDelete,    Icon: TrashIcon, danger: true },
  ];

  return (
    <div style={{
      position: "absolute", top: "calc(100% + 8px)", right: 0, zIndex: 40,
      minWidth: "240px",
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: "14px", padding: "6px",
      boxShadow: "0 12px 32px rgba(0,0,0,0.35)",
      animation: "fadeUp 0.15s ease forwards",
    }}>
      {items.map((item, i) => {
        if (item.type === "separator") {
          return <div key={`sep-${i}`} style={{ height: "1px", background: "var(--border)", margin: "6px 8px" }} />;
        }

        const isLink = item.type === "link";
        const Icon   = item.Icon;
        const color  = item.danger ? "#e87070" : (isLink ? "var(--foreground)" : "var(--foreground)");

        const commonStyle: React.CSSProperties = {
          display: "flex", alignItems: "center", gap: "10px",
          width: "100%", padding: "9px 12px",
          borderRadius: "10px", fontSize: "13px",
          background: "transparent", border: "none",
          color, cursor: "pointer", textAlign: "left",
          textDecoration: "none",
          transition: "background 0.15s",
        };
        const onEnter = (e: React.MouseEvent<HTMLElement>) => {
          e.currentTarget.style.background = item.danger ? "rgba(232,112,112,0.08)" : "rgba(255,255,255,0.04)";
        };
        const onLeave = (e: React.MouseEvent<HTMLElement>) => {
          e.currentTarget.style.background = "transparent";
        };

        if (isLink) {
          const href = item.href;
          const isExternal = href.endsWith(".html");
          return (
            <a
              key={item.label}
              href={href}
              target={isExternal ? "_blank" : undefined}
              rel={isExternal ? "noopener noreferrer" : undefined}
              onClick={(e) => {
                if (isExternal) return;
                e.preventDefault();
                onNavigate(href);
              }}
              onMouseEnter={onEnter}
              onMouseLeave={onLeave}
              style={commonStyle}
            >
              <Icon size={15} />
              <span>{item.label}</span>
            </a>
          );
        }

        return (
          <button
            key={item.label}
            onClick={item.onClick}
            onMouseEnter={onEnter}
            onMouseLeave={onLeave}
            style={commonStyle}
          >
            <Icon size={15} />
            <span>{item.label}</span>
          </button>
        );
      })}
    </div>
  );
}

// ── ChangePasswordModal ──────────────────────────────────────────────────────

function ChangePasswordModal({ onClose, onSuccess }: { onClose: () => void; onSuccess: () => void }) {
  const [oldPwd, setOldPwd]   = useState("");
  const [newPwd, setNewPwd]   = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError]     = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const strength = checkPasswordStrength(newPwd);
  const allGood  = strength.length && strength.uppercase && strength.digit;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newPwd !== confirm) { setError("Les mots de passe ne correspondent pas"); return; }
    if (!allGood) { setError("Le mot de passe ne respecte pas les critères"); return; }
    setError(null); setLoading(true);
    try {
      await changePassword(oldPwd, newPwd);
      onSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur");
      setLoading(false);
    }
  };

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 100, display: "flex",
      alignItems: "center", justifyContent: "center",
      background: "rgba(12,12,14,0.85)", backdropFilter: "blur(4px)",
    }} onClick={onClose}>
      <div style={{
        width: "100%", maxWidth: "380px", margin: "24px",
        background: "var(--surface)", border: "1px solid var(--border)",
        borderRadius: "20px", padding: "28px", animation: "fadeUp 0.3s ease",
      }} onClick={e => e.stopPropagation()}>
        <h2 className="font-serif" style={{ fontSize: "20px", color: "var(--foreground)", marginBottom: "24px" }}>
          Changer le mot de passe
        </h2>
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
          {[
            { label: "Ancien mot de passe", value: oldPwd, set: setOldPwd },
            { label: "Nouveau mot de passe", value: newPwd, set: setNewPwd },
            { label: "Confirmer",            value: confirm, set: setConfirm },
          ].map(f => (
            <div key={f.label} style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <label style={{ fontSize: "11px", letterSpacing: "0.15em", textTransform: "uppercase", color: "var(--muted)" }}>{f.label}</label>
              <input type="password" value={f.value} onChange={e => f.set(e.target.value)} placeholder="••••••••" required style={{
                padding: "11px 14px", borderRadius: "10px", fontSize: "14px",
                background: "var(--background)", border: "1px solid var(--border)",
                color: "var(--foreground)", outline: "none", width: "100%",
              }} />
            </div>
          ))}
          {newPwd.length > 0 && (
            <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
              {[{ ok: strength.length, label: "8 car." }, { ok: strength.uppercase, label: "Maj." }, { ok: strength.digit, label: "Chiffre" }].map(r => (
                <span key={r.label} style={{ fontSize: "11px", color: r.ok ? "#7ec9a0" : "var(--muted)" }}>
                  {r.ok ? "✓" : "○"} {r.label}
                </span>
              ))}
            </div>
          )}
          {error && (
            <p style={{ fontSize: "13px", color: "#e87070", padding: "10px 14px", borderRadius: "10px", background: "rgba(232,112,112,0.08)", border: "1px solid rgba(232,112,112,0.2)" }}>
              {error}
            </p>
          )}
          <div style={{ display: "flex", gap: "10px", marginTop: "4px" }}>
            <button type="button" onClick={onClose} style={{
              flex: 1, padding: "11px", borderRadius: "10px", cursor: "pointer", fontSize: "13px",
              background: "none", border: "1px solid var(--border)", color: "var(--muted)",
            }}>Annuler</button>
            <button type="submit" disabled={loading} style={{
              flex: 2, padding: "11px", borderRadius: "10px", cursor: loading ? "wait" : "pointer",
              fontSize: "13px", fontWeight: 500, border: "none",
              background: loading ? "var(--border)" : "linear-gradient(135deg, var(--accent), var(--accent-dim))",
              color: loading ? "var(--muted)" : "#0c0c0e",
            }}>{loading ? "Enregistrement…" : "Enregistrer"}</button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ── DeleteAccountModal ──────────────────────────────────────────────────────

function DeleteAccountModal({ onClose, onDeleted }: { onClose: () => void; onDeleted: () => void }) {
  const [loading, setLoading] = useState(false);

  const handleDelete = async () => {
    setLoading(true);
    try { await deleteAccount(); onDeleted(); }
    catch { setLoading(false); }
  };

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 100, display: "flex",
      alignItems: "center", justifyContent: "center",
      background: "rgba(12,12,14,0.85)", backdropFilter: "blur(4px)",
    }} onClick={onClose}>
      <div style={{
        width: "100%", maxWidth: "360px", margin: "24px",
        background: "var(--surface)", border: "1px solid rgba(232,112,112,0.3)",
        borderRadius: "20px", padding: "28px", animation: "fadeUp 0.3s ease",
      }} onClick={e => e.stopPropagation()}>
        <h2 className="font-serif" style={{ fontSize: "20px", color: "var(--foreground)", marginBottom: "12px" }}>
          Supprimer le compte
        </h2>
        <p style={{ fontSize: "13px", color: "var(--muted)", marginBottom: "24px", lineHeight: 1.6 }}>
          Cette action est irréversible. Toutes vos données seront supprimées définitivement.
        </p>
        <div style={{ display: "flex", gap: "10px" }}>
          <button onClick={onClose} style={{
            flex: 1, padding: "11px", borderRadius: "10px", cursor: "pointer",
            background: "none", border: "1px solid var(--border)", color: "var(--muted)", fontSize: "13px",
          }}>Annuler</button>
          <button onClick={handleDelete} disabled={loading} style={{
            flex: 2, padding: "11px", borderRadius: "10px", cursor: loading ? "wait" : "pointer",
            background: "rgba(232,112,112,0.15)", border: "1px solid rgba(232,112,112,0.3)",
            color: "#e87070", fontSize: "13px", fontWeight: 500,
          }}>{loading ? "Suppression…" : "Supprimer définitivement"}</button>
        </div>
      </div>
    </div>
  );
}

// ── Utils ────────────────────────────────────────────────────────────────────

function detectCurrent(pathname: string): Current {
  if (pathname.startsWith("/translate")) return "translate";
  if (pathname.startsWith("/live"))      return "live";
  if (pathname.startsWith("/meeting"))   return "meeting";
  if (pathname.startsWith("/admin"))     return "admin";
  return null;
}
