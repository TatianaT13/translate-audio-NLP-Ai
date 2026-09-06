const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8004";

export interface User {
  id: number;
  email: string;
  is_admin: boolean;
  created_at: string;
}

// ── Token storage ─────────────────────────────────────────────────────────────

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("access_token");
}

export function setTokens(accessToken: string, refreshToken: string) {
  localStorage.setItem("access_token", accessToken);
  localStorage.setItem("refresh_token", refreshToken);
  // Cookie pour le middleware Next.js — aligné sur le refresh token (7j).
  // La vraie validation JWT reste faite par le gateway ; le cookie ne sert
  // qu'à savoir "présence de session" pour l'auth-gate côté SSR/middleware.
  document.cookie = `access_token=${accessToken}; path=/; max-age=604800; SameSite=Strict`;
}

export function clearTokens() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
  document.cookie = "access_token=; path=/; max-age=0";
}

// ── Auth calls ────────────────────────────────────────────────────────────────

export async function register(email: string, password: string) {
  const res = await fetch(`${GATEWAY_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Erreur lors de la création du compte");
  }
  return res.json();
}

export async function login(email: string, password: string) {
  const res = await fetch(`${GATEWAY_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Email ou mot de passe incorrect");
  }
  const data = await res.json();
  setTokens(data.access_token, data.refresh_token);
  return data;
}

export async function logout() {
  const refreshToken = localStorage.getItem("refresh_token");
  const accessToken  = getAccessToken();
  if (refreshToken && accessToken) {
    await fetch(`${GATEWAY_URL}/auth/logout`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${accessToken}`,
      },
      body: JSON.stringify({ refresh_token: refreshToken }),
    }).catch(() => {});
  }
  clearTokens();
}

export async function refreshAccessToken(): Promise<string | null> {
  const refreshToken = localStorage.getItem("refresh_token");
  if (!refreshToken) return null;

  const res = await fetch(`${GATEWAY_URL}/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });
  if (!res.ok) {
    clearTokens();
    return null;
  }
  const data = await res.json();
  setTokens(data.access_token, data.refresh_token);
  return data.access_token;
}

export async function getMe(): Promise<User | null> {
  let token = getAccessToken();
  if (!token) return null;
  let res = await fetch(`${GATEWAY_URL}/auth/me`, {
    headers: { "Authorization": `Bearer ${token}` },
  });
  // Access token expiré (JWT 15min) — tente un refresh silencieux avec
  // le refresh token (7j) avant d'abandonner. Sinon le menu utilisateur
  // disparaît sans raison visible pour l'user encore actif.
  if (res.status === 401) {
    token = await refreshAccessToken();
    if (!token) return null;
    res = await fetch(`${GATEWAY_URL}/auth/me`, {
      headers: { "Authorization": `Bearer ${token}` },
    });
  }
  if (!res.ok) return null;
  return res.json();
}

// Appelle une URL avec Bearer JWT. En cas de 401, refresh silencieux et retry.
// À utiliser depuis les composants pour tout call vers Gateway ou Pipeline.
export async function authFetch(url: string, init: RequestInit = {}): Promise<Response> {
  let token = getAccessToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  let res = await fetch(url, { ...init, headers });
  if (res.status === 401) {
    token = await refreshAccessToken();
    if (!token) return res;
    headers.set("Authorization", `Bearer ${token}`);
    res = await fetch(url, { ...init, headers });
  }
  return res;
}

export async function changePassword(oldPassword: string, newPassword: string) {
  const token = getAccessToken();
  const res = await fetch(`${GATEWAY_URL}/auth/change-password`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`,
    },
    body: JSON.stringify({ old_password: oldPassword, new_password: newPassword }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Erreur lors du changement de mot de passe");
  }
  clearTokens();
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

export async function deleteAccount() {
  const token = getAccessToken();
  const res = await fetch(`${GATEWAY_URL}/auth/account`, {
    method: "DELETE",
    headers: { "Authorization": `Bearer ${token}` },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Erreur lors de la suppression du compte");
  }
  clearTokens();
}

// ── Password strength ─────────────────────────────────────────────────────────

export function checkPasswordStrength(password: string) {
  return {
    length:    password.length >= 8,
    uppercase: /[A-Z]/.test(password),
    digit:     /[0-9]/.test(password),
  };
}
