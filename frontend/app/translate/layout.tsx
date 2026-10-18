import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Traduire un audio",
  description:
    "Importez un fichier audio ou enregistrez-vous : transcription, traduction en 30 langues et voix de synthese fidele au locuteur original.",
  alternates: { canonical: "/translate" },
  openGraph: {
    title: "Traduire un audio — traduction-audio.fr",
    description: "Transcription, traduction et voix clonee en quelques secondes.",
    url: "/translate",
  },
};

export default function TranslateLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
