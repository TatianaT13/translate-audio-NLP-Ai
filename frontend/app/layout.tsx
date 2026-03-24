import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "traduction-audio.fr — Translate any voice, instantly",
  description: "Upload or record audio in French and get an instant English translation powered by AI.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="fr" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
