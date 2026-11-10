const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8004";

export interface User {
  id: number;
  email: string;
  is_admin: boolean;
  /** false tant que l'adresse n'est pas confirmée : crédits offerts en attente. */
  email_verified?: boolean;
  created_at: string;
}

// ── Session ───────────────────────────────────────────────────────────────────
// Les jetons vivent dans des cookies HttpOnly posés par la Gateway : le
// JavaScript de la page ne peut pas les lire, donc un script injecté non plus.
// Seul le témoin `session` (aucun secret) est lisible ici, pour savoir s'il
// vaut la peine d'interroger l'API.

function hasSession(): boolean {
  return typeof document !== "undefined" && /(?:^|;\s*)session=1/.test(document.cookie);
}

function clearSessionMarker() {
  document.cookie = "session=; path=/; max-age=0";
}

/** Efface les jetons que les anciennes versions du site stockaient dans le navigateur. */
function dropLegacyTokens() {
  try {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
  } catch {}
}

// ── Auth calls ────────────────────────────────────────────────────────────────

export async function register(email: string, password: string, turnstileToken?: string | null) {
  const res = await fetch(`${GATEWAY_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, turnstile_token: turnstileToken ?? null }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Erreur lors de la création du compte");
  }
  return res.json();
}

export async function login(email: string, password: string): Promise<void> {
  const res = await fetch(`${GATEWAY_URL}/auth/login`, {
    method: "POST",
    credentials: "include",   // la réponse pose les cookies de session
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Email ou mot de passe incorrect");
  }
  dropLegacyTokens();
}

export async function logout() {
  await fetch(`${GATEWAY_URL}/auth/logout`, { method: "POST", credentials: "include" }).catch(() => {});
  dropLegacyTokens();
  clearSessionMarker();   // même si le serveur n'a pas répondu
}

let refreshInFlight: Promise<boolean> | null = null;

// La Gateway invalide l'ancien refresh token à chaque rotation : deux refresh
// simultanés feraient échouer le second. Les appels concurrents partagent donc
// le même refresh.
export function refreshAccessToken(): Promise<boolean> {
  if (!refreshInFlight) {
    refreshInFlight = doRefresh().finally(() => { refreshInFlight = null; });
  }
  return refreshInFlight;
}

async function doRefresh(): Promise<boolean> {
  try {
    const res = await fetch(`${GATEWAY_URL}/auth/refresh`, { method: "POST", credentials: "include" });
    return res.ok;
  } catch {
    return false;
  }
}

export async function getMe(): Promise<User | null> {
  if (!hasSession()) return null;   // visiteur non connecté : pas d'appel inutile
  const res = await authFetch(`${GATEWAY_URL}/auth/me`);
  if (!res.ok) return null;
  return res.json();
}

/**
 * fetch authentifié par les cookies de session. Sur un 401 (access token
 * expiré, 15 min), renouvelle la session puis rejoue la requête une fois.
 */
export async function authFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const opts: RequestInit = { ...init, credentials: "include" };
  let res = await fetch(url, opts);
  if (res.status === 401 && hasSession()) {
    const renewed = await refreshAccessToken();
    // On rejoue même si le refresh a échoué : un autre onglet a pu renouveler
    // la session entre-temps, et ses cookies sont partagés.
    res = await fetch(url, opts);
    if (res.status === 401 && !renewed) clearSessionMarker();   // session terminée
  }
  return res;
}

export async function changePassword(oldPassword: string, newPassword: string) {
  const res = await authFetch(`${GATEWAY_URL}/auth/change-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ old_password: oldPassword, new_password: newPassword }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Erreur lors du changement de mot de passe");
  }
  clearSessionMarker();   // la Gateway a fermé la session : il faut se reconnecter
  return res.json();
}

export async function forgotPassword(email: string) {
  const res = await fetch(`${GATEWAY_URL}/auth/forgot-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Erreur");
  }
  return res.json();
}

export async function resetPassword(token: string, newPassword: string) {
  const res = await fetch(`${GATEWAY_URL}/auth/reset-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, new_password: newPassword }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Lien invalide ou expiré");
  }
  return res.json();
}

export async function verifyEmail(token: string): Promise<{ message: string }> {
  const res = await fetch(`${GATEWAY_URL}/auth/verify-email`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Lien invalide ou expiré");
  }
  return res.json();
}

export async function resendVerification(): Promise<{ message: string }> {
  const res = await authFetch(`${GATEWAY_URL}/auth/resend-verification`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(res.status === 429 ? "Trop de demandes. Réessayez dans une heure." : err.detail || "Envoi impossible");
  }
  return res.json();
}

export async function deleteAccount() {
  const res = await authFetch(`${GATEWAY_URL}/auth/account`, { method: "DELETE" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Erreur lors de la suppression du compte");
  }
  dropLegacyTokens();
  clearSessionMarker();
}

// ── Password strength ─────────────────────────────────────────────────────────

export function checkPasswordStrength(password: string) {
  return {
    length:    password.length >= 8,
    uppercase: /[A-Z]/.test(password),
    digit:     /[0-9]/.test(password),
  };
}
