import { LegalLayout } from "@/components/LegalLayout";

export const metadata = { title: "Mentions légales — traduction-audio.fr" };

export default function MentionsLegales() {
  return (
    <LegalLayout title="Mentions légales" updated="15 septembre 2026">
      <p>
        Conformément aux dispositions de la <strong>loi n° 2004-575 du 21 juin 2004</strong> pour
        la confiance dans l&apos;économie numérique (LCEN), il est précisé aux utilisateurs du
        site <strong>traduction-audio.fr</strong> l&apos;identité des différents intervenants dans le
        cadre de sa réalisation et de son suivi.
      </p>

      <h2>1. Éditeur du site</h2>
      <dl>
        <dt>Éditeur</dt>
        <dd>Tetyana Tarasenko</dd>
        <dt>Forme juridique</dt>
        <dd>Entrepreneur individuel</dd>
        <dt>Adresse</dt>
        <dd>5A Rue des Argillières, 21121 Ahuy, France</dd>
        <dt>Numéro SIRET</dt>
        <dd>838 177 012 00021</dd>
        <dt>Code APE</dt>
        <dd>6201Z — Programmation informatique</dd>
        <dt>Numéro de TVA intracommunautaire</dt>
        <dd>Non applicable — franchise en base de TVA (article 293 B du Code général des impôts)</dd>
        <dt>Directeur de la publication</dt>
        <dd>Tetyana Tarasenko</dd>
        <dt>Contact</dt>
        <dd><a href="/contact">Formulaire de contact</a></dd>
      </dl>

      <h2>2. Hébergement</h2>
      <dl>
        <dt>Hébergeur</dt>
        <dd>Hetzner Online GmbH</dd>
        <dt>Adresse</dt>
        <dd>Industriestr. 25, 91710 Gunzenhausen, Allemagne</dd>
        <dt>Contact hébergeur</dt>
        <dd>+49 (0)9831 505-0 — <a href="https://www.hetzner.com" target="_blank" rel="noopener noreferrer">www.hetzner.com</a></dd>
      </dl>

      <h2>3. Propriété intellectuelle</h2>
      <p>
        L&apos;ensemble des éléments présents sur ce site (textes, graphismes, logos, icônes, images,
        vidéos, sons, logiciels) est la propriété exclusive de l&apos;éditeur ou fait l&apos;objet
        d&apos;une autorisation d&apos;utilisation. Toute reproduction, représentation, modification,
        publication, adaptation, totale ou partielle, quel que soit le moyen ou le procédé utilisé,
        est interdite sans l&apos;autorisation écrite préalable de l&apos;éditeur, sous peine de
        constituer une contrefaçon au sens des articles L.335-2 et suivants du Code de la propriété
        intellectuelle.
      </p>

      <h2>4. Responsabilité</h2>
      <p>
        L&apos;éditeur met tout en œuvre pour offrir aux utilisateurs des informations et outils
        disponibles et vérifiés, mais ne saurait être tenu pour responsable des erreurs ou d&apos;une
        indisponibilité du service. L&apos;utilisation du service se fait sous la seule responsabilité
        de l&apos;utilisateur.
      </p>

      <h2>5. Droit applicable</h2>
      <p>
        Le présent site est soumis au droit français. En cas de litige, et à défaut de résolution
        amiable, compétence est attribuée aux tribunaux français compétents.
      </p>

      <h2>6. Signalement d&apos;un contenu illicite</h2>
      <p>
        Conformément à l&apos;article 6-I-5 de la LCEN, tout signalement de contenu manifestement
        illicite peut être adressé via <a href="/contact">notre formulaire de contact</a>.
      </p>
    </LegalLayout>
  );
}
