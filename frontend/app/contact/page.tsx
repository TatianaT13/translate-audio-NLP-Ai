import { LegalLayout } from "@/components/LegalLayout";

export const metadata = { title: "Contact — traduction-audio.fr" };

export default function Contact() {
  return (
    <LegalLayout title="Contact" updated="15 septembre 2026">
      <p>
        Une question, une remarque, une demande de partenariat ? Nous sommes à votre écoute.
        Choisissez le canal le plus adapté à votre demande.
      </p>

      <h2>Support utilisateur</h2>
      <p>
        Pour toute question technique, aide à l&apos;utilisation ou signalement de bug :
      </p>
      <dl>
        <dt>Email</dt>
        <dd><a href="mailto:support@traduction-audio.fr">support@traduction-audio.fr</a></dd>
        <dt>Délai de réponse</dt>
        <dd>Sous 48 h ouvrées</dd>
      </dl>

      <h2>Facturation et compte</h2>
      <p>
        Pour toute question relative à votre abonnement, une facture ou la gestion de votre compte :
      </p>
      <dl>
        <dt>Email</dt>
        <dd><a href="mailto:facturation@traduction-audio.fr">facturation@traduction-audio.fr</a></dd>
      </dl>

      <h2>Données personnelles (RGPD)</h2>
      <p>
        Pour exercer vos droits sur vos données personnelles (accès, rectification, suppression,
        portabilité) :
      </p>
      <dl>
        <dt>Email</dt>
        <dd><a href="mailto:privacy@traduction-audio.fr">privacy@traduction-audio.fr</a></dd>
        <dt>Voir aussi</dt>
        <dd><a href="/confidentialite">Politique de confidentialité</a></dd>
      </dl>

      <h2>Presse et partenariats</h2>
      <p>
        Pour toute demande professionnelle :
      </p>
      <dl>
        <dt>Email</dt>
        <dd><a href="mailto:contact@traduction-audio.fr">contact@traduction-audio.fr</a></dd>
      </dl>

      <h2>Adresse postale</h2>
      <dl>
        <dt>Éditeur</dt>
        <dd>Tetyana Tarasenko — entrepreneur individuel</dd>
        <dt>Adresse</dt>
        <dd>5A Rue des Argillières, 21121 Ahuy, France</dd>
      </dl>

      <p style={{ marginTop: "40px", fontStyle: "italic" }}>
        Un formulaire de contact interactif est en préparation. En attendant, merci d&apos;utiliser
        l&apos;email correspondant à votre demande.
      </p>
    </LegalLayout>
  );
}
