import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Tarifs et abonnements",
  description:
    "Pro à 9,90 € par mois ou 99 € par an. 10 crédits offerts à l'inscription, sans carte bancaire. Sans engagement.",
  alternates: { canonical: "/tarifs" },
  openGraph: {
    title: "Tarifs — traduction-audio.fr",
    description: "Pro à 9,90 € par mois. 10 crédits offerts à l'inscription.",
    url: "/tarifs",
  },
};

export default function TarifsLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
