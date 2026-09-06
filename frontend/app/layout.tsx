import type { Metadata } from "next";
import { Special_Elite, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const specialElite = Special_Elite({
  subsets: ["latin"],
  weight: ["400"],
  display: "swap",
  variable: "--font-display",
});

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
  style: ["normal", "italic"],
  display: "swap",
  variable: "--font-body",
});

export const metadata: Metadata = {
  title: "traduction-audio.fr — Translate any voice, instantly",
  description: "Upload or record audio in French and get an instant English translation powered by AI.",
  // Le favicon passe par /api/favicon.ico (exempté de basic auth nginx via rewrite Next.js)
  // pour être servi avant même que l'user ne soit authentifié.
  icons: {
    icon: "/api/favicon.ico",
    shortcut: "/api/favicon.ico",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="fr" className={`${specialElite.variable} ${ibmPlexMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
