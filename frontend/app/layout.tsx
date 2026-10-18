import type { Metadata } from "next";
import { Special_Elite, IBM_Plex_Mono } from "next/font/google";
import { Analytics } from "@/components/Analytics";
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

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://traduction-audio.fr";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "traduction-audio.fr — Transcription et traduction audio par IA",
    template: "%s — traduction-audio.fr",
  },
  description:
    "Transcrivez, traduisez et resumez vos audios en 30 langues. Identification des intervenants, voix clonee, comptes-rendus de reunion. Essayez gratuitement.",
  keywords: [
    "transcription audio", "traduction audio", "intelligence artificielle",
    "whisper", "compte-rendu de reunion", "diarisation", "sous-titres",
    "podcast traduction", "voix IA", "SaaS francais",
  ],
  authors: [{ name: "Tetyana Tarasenko" }],
  creator: "Tetyana Tarasenko",
  publisher: "traduction-audio.fr",
  alternates: { canonical: "/" },
  robots: {
    index: true, follow: true,
    googleBot: {
      index: true, follow: true,
      "max-video-preview": -1, "max-image-preview": "large", "max-snippet": -1,
    },
  },
  openGraph: {
    type: "website",
    locale: "fr_FR",
    url: SITE_URL,
    siteName: "traduction-audio.fr",
    title: "traduction-audio.fr — Transcription et traduction audio par IA",
    description:
      "Traduisez vos audios en 30 langues. Diarisation, voix clonee, comptes-rendus de reunion. Essayez gratuitement.",
    // L'image OG est generee automatiquement par app/opengraph-image.tsx (1200x630).
  },
  twitter: {
    card: "summary_large_image",
    title: "traduction-audio.fr — Transcription et traduction audio par IA",
    description: "Traduisez vos audios en 30 langues. Essayez gratuitement.",
  },
  icons: {
    icon: "/api/favicon.ico",
    shortcut: "/api/favicon.ico",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="fr" className={`${specialElite.variable} ${ibmPlexMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        {children}
        <Analytics />
      </body>
    </html>
  );
}
