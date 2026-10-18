import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Traduire un audio",
  description:
    "Importez un fichier audio en français ou enregistrez-vous : transcription, traduction en anglais, espagnol, allemand ou ukrainien, et voix de synthèse.",
  alternates: { canonical: "/translate" },
  openGraph: {
    title: "Traduire un audio — traduction-audio.fr",
    description: "Transcription, traduction et voix de synthèse.",
    url: "/translate",
  },
};

export default function TranslateLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
