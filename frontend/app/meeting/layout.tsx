import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Comptes-rendus de réunion",
  description:
    "Enregistrez vos réunions, identifiez les intervenants et générez un compte-rendu structuré, exportable en PDF.",
  alternates: { canonical: "/meeting" },
  openGraph: {
    title: "Comptes-rendus de réunion — traduction-audio.fr",
    description: "Identification des intervenants, résumé, export PDF et SRT.",
    url: "/meeting",
  },
};

export default function MeetingLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
