import { LegalLayout } from "@/components/LegalLayout";

export const metadata = { title: "Gestion des cookies — traduction-audio.fr" };

export default function Cookies() {
  return (
    <LegalLayout title="Gestion des cookies" updated="15 septembre 2026">
      <p>
        Le site <strong>traduction-audio.fr</strong> utilise des cookies et technologies similaires
        pour assurer son bon fonctionnement, mesurer son audience et améliorer votre expérience.
        Cette page vous informe sur les cookies déposés et vos choix.
      </p>

      <h2>1. Qu&apos;est-ce qu&apos;un cookie ?</h2>
      <p>
        Un cookie est un petit fichier texte déposé sur votre terminal (ordinateur, tablette,
        smartphone) lors de la visite d&apos;un site web. Il permet au site de reconnaître votre
        navigateur, de mémoriser vos préférences ou de suivre votre activité.
      </p>

      <h2>2. Cookies utilisés</h2>

      <h3>Cookies strictement nécessaires (sans consentement)</h3>
      <p>Ces cookies sont indispensables au fonctionnement du Service et ne peuvent être désactivés :</p>
      <ul>
        <li><strong>access_token</strong> — authentification de session, durée : 7 jours</li>
        <li><strong>refresh_token</strong> (localStorage) — renouvellement du token d&apos;accès, durée : 7 jours</li>
        <li><strong>préférences utilisateur</strong> (localStorage) — langue cible, mode de vitesse audio, etc.</li>
      </ul>

      <h3>Cookies de mesure d&apos;audience</h3>
      <p>
        Aucun outil de mesure d&apos;audience n&apos;est actuellement déployé sur le site.
        Cette section sera mise à jour si un outil respectueux de la vie privée
        (Plausible, Matomo) venait à être ajouté.
      </p>

      <h3>Cookies tiers</h3>
      <p>
        Certains services tiers peuvent déposer leurs propres cookies dans le cadre de leur
        fonctionnement. Ces cookies sont soumis à la politique de confidentialité de leurs
        éditeurs respectifs :
      </p>
      <ul>
        <li><strong>Stripe</strong> (paiement) — <a href="https://stripe.com/fr/privacy" target="_blank" rel="noopener noreferrer">stripe.com/fr/privacy</a></li>
      </ul>

      <h2>3. Gérer vos préférences</h2>
      <p>
        Vous pouvez à tout moment modifier vos préférences en matière de cookies :
      </p>
      <ul>
        <li>Depuis la bannière de gestion des cookies affichée lors de votre première visite</li>
        <li>Depuis les paramètres de votre navigateur, en supprimant tout ou partie des cookies stockés</li>
      </ul>
      <p>
        <strong>Attention</strong> : la désactivation des cookies strictement nécessaires empêchera
        le bon fonctionnement du Service, notamment l&apos;authentification.
      </p>

      <h2>4. Comment désactiver les cookies dans votre navigateur ?</h2>
      <ul>
        <li><strong>Chrome</strong> : Paramètres → Confidentialité et sécurité → Cookies et autres données des sites</li>
        <li><strong>Firefox</strong> : Paramètres → Vie privée et sécurité → Cookies et données de sites</li>
        <li><strong>Safari</strong> : Préférences → Confidentialité → Gérer les données de site web</li>
        <li><strong>Edge</strong> : Paramètres → Cookies et autorisations de site</li>
      </ul>

      <h2>5. Durée de validité de votre consentement</h2>
      <p>
        Votre consentement est valable pour une durée maximale de <strong>13 mois</strong>, à
        compter de sa dernière expression. À l&apos;issue de cette période, votre consentement vous
        sera à nouveau demandé.
      </p>

      <h2>6. En savoir plus</h2>
      <p>
        Pour en savoir plus sur les cookies et la protection de votre vie privée, consultez le site
        de la CNIL : <a href="https://www.cnil.fr/fr/cookies-et-traceurs" target="_blank" rel="noopener noreferrer">www.cnil.fr/fr/cookies-et-traceurs</a>.
      </p>
    </LegalLayout>
  );
}
