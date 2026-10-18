import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Tarifs et abonnements",
  description:
    "Formules a partir de 9 EUR/mois. Essai gratuit avec 10 credits inclus. Pas d'engagement, resiliation en un clic.",
  alternates: { canonical: "/tarifs" },
  openGraph: {
    title: "Tarifs — traduction-audio.fr",
    description: "Formules a partir de 9 EUR/mois. Essai gratuit.",
    url: "/tarifs",
  },
};

export default function TarifsLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
