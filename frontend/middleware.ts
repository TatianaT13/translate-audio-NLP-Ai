import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const PUBLIC_PATHS  = ["/login", "/register", "/forgot-password", "/reset-password"];
// Préfixes proxyés vers les backends Docker → laisse passer sans auth Next.js
// (l'auth réelle est gérée par le service cible, ex: JWT côté gateway)
const PROXY_PREFIXES = ["/api", "/pipeline", "/stt", "/llm"];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Laisser passer les assets statiques (tout ce qui contient un point dans le nom)
  // ex: /demo.mp3, /favicon.ico, /logo.png
  const lastSegment = pathname.split("/").pop() ?? "";
  if (lastSegment.includes(".")) {
    return NextResponse.next();
  }

  // Laisse passer les rewrites vers les backends (gateway/pipeline/stt/llm)
  if (PROXY_PREFIXES.some((p) => pathname.startsWith(p))) {
    return NextResponse.next();
  }

  // Allow public auth routes
  if (PUBLIC_PATHS.some((p) => pathname.startsWith(p))) {
    return NextResponse.next();
  }

  // Check access_token cookie (set by JS after login)
  const token = request.cookies.get("access_token")?.value;

  if (!token) {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  return NextResponse.next();
}

export const config = {
  // Exclure _next, favicon, et toute URL contenant un point (= asset statique)
  // Le motif "[^.]*$" matche uniquement les chemins SANS point → seules les pages app passent par l'auth
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)", "/"],
};

// Note : on filtre dans le middleware lui-même les requêtes vers des fichiers
// statiques (extension dans le pathname) pour éviter les 400 sur /demo.mp3
