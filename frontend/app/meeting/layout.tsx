import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Comptes-rendus de reunion",
  description:
    "Enregistrez vos reunions, identifiez les intervenants et generez un compte-rendu structure dans la langue de votre choix.",
  alternates: { canonical: "/meeting" },
  openGraph: {
    title: "Comptes-rendus de reunion — traduction-audio.fr",
    description: "Diarisation, resume executif, notes par intervenant.",
    url: "/meeting",
  },
};

export default function MeetingLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
