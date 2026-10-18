import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Traduction en direct",
  description:
    "Parlez : vos paroles sont transcrites, traduites et vocalisees en temps reel. Ideal pour conversations multilingues et conferences.",
  alternates: { canonical: "/live" },
  openGraph: {
    title: "Traduction en direct — traduction-audio.fr",
    description: "Transcription, traduction et voix en temps reel.",
    url: "/live",
  },
};

export default function LiveLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
