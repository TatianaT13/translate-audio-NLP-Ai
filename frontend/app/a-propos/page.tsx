import { LegalLayout } from "@/components/LegalLayout";

export const metadata = { title: "À propos — traduction-audio.fr" };

export default function APropos() {
  return (
    <LegalLayout title="À propos" updated="15 septembre 2026">
      <p>
        <strong>traduction-audio.fr</strong> est une plateforme française de traduction audio à la
        demande, pensée pour rendre l&apos;information accessible à toutes celles et ceux qui
        franchissent des barrières de langue au quotidien.
      </p>

      <h2>Notre mission</h2>
      <p>
        Faciliter la communication multilingue en offrant un service simple, rapide et fiable pour
        traduire des contenus audio dans plusieurs langues, sans expertise technique requise.
      </p>

      <h2>Nos valeurs</h2>
      <ul>
        <li><strong>Simplicité</strong> — un outil accessible, sans jargon technique</li>
        <li><strong>Confidentialité</strong> — vos fichiers audio ne sont jamais conservés</li>
        <li><strong>Transparence</strong> — nos limites sont clairement affichées, nos coûts prévisibles</li>
        <li><strong>Souveraineté</strong> — infrastructure hébergée en Europe, dans le respect du RGPD</li>
      </ul>

      <h2>Nos cas d&apos;usage</h2>
      <ul>
        <li>Rendre accessibles des annonces sonores à un public multilingue</li>
        <li>Faciliter la communication en mobilité (voyages, échanges professionnels)</li>
        <li>Produire des comptes-rendus de réunion clairs et exploitables</li>
      </ul>

      <h2>L&apos;équipe</h2>
      <p>
        Le projet est né en 2026 à l&apos;initiative de <strong>Tetyana Tarasenko</strong>, dans le
        cadre d&apos;un travail de fin d&apos;études en ingénierie IA. Aujourd&apos;hui, il évolue
        vers une plateforme ouverte aux utilisateurs professionnels et grand public.
      </p>

      <h2>Nous rejoindre</h2>
      <p>
        Vous êtes intéressé par notre démarche, un partenariat ou une collaboration ? Écrivez-nous
        via <a href="/contact">notre formulaire de contact</a>.
      </p>
    </LegalLayout>
  );
}
