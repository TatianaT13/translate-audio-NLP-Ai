import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const PUBLIC_PATHS  = ["/login", "/register", "/forgot-password", "/reset-password"];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

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
  // Exclure tous les assets statiques + extensions de fichiers du middleware d'auth
  matcher: ["/((?!_next/static|_next/image|favicon.ico|.*\\.(?:mp3|wav|m4a|ogg|svg|png|jpg|jpeg|gif|webp|ico)$).*)"],
};
