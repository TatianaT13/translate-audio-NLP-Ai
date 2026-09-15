import { LegalLayout } from "@/components/LegalLayout";

export const metadata = { title: "Conditions générales d'utilisation — traduction-audio.fr" };

export default function CGU() {
  return (
    <LegalLayout title="Conditions générales d'utilisation" updated="15 septembre 2026">
      <p>
        Les présentes Conditions générales d&apos;utilisation (« <strong>CGU</strong> »)
        encadrent l&apos;utilisation du site <strong>traduction-audio.fr</strong> (le « Service ») édité
        par <span className="placeholder">[NOM DE L&apos;ÉDITEUR]</span>. En créant un compte ou en
        utilisant le Service, vous acceptez sans réserve les présentes CGU.
      </p>

      <h2>1. Objet</h2>
      <p>
        Le Service permet à ses utilisateurs de transcrire, traduire et synthétiser des contenus audio
        dans plusieurs langues, dans le cadre d&apos;un usage personnel ou professionnel.
      </p>

      <h2>2. Accès au Service</h2>
      <p>
        L&apos;accès aux fonctionnalités du Service requiert la création d&apos;un compte
        utilisateur, gratuit ou payant selon la formule choisie. L&apos;utilisateur s&apos;engage à
        fournir des informations exactes lors de son inscription et à les tenir à jour.
      </p>
      <p>
        L&apos;utilisateur est seul responsable de la confidentialité de ses identifiants et de toute
        activité effectuée depuis son compte. Toute utilisation frauduleuse doit être signalée sans
        délai via le formulaire de contact.
      </p>

      <h2>3. Usage acceptable</h2>
      <p>L&apos;utilisateur s&apos;engage à ne pas utiliser le Service pour :</p>
      <ul>
        <li>Diffuser des contenus illicites, diffamatoires, haineux, discriminatoires ou contraires aux bonnes mœurs</li>
        <li>Porter atteinte aux droits de tiers, notamment aux droits d&apos;auteur, à la vie privée ou à l&apos;image</li>
        <li>Contourner les limites techniques du Service, tenter d&apos;en altérer le fonctionnement ou d&apos;en extraire des données de manière automatisée</li>
        <li>Utiliser le Service à des fins de spam, de phishing ou de toute autre activité malveillante</li>
      </ul>
      <p>
        En cas de manquement, l&apos;éditeur se réserve le droit de suspendre ou de supprimer le
        compte, sans préavis et sans indemnisation.
      </p>

      <h2>4. Propriété du contenu utilisateur</h2>
      <p>
        L&apos;utilisateur conserve la propriété des contenus qu&apos;il soumet au Service (fichiers
        audio, textes). En les téléversant, il concède à l&apos;éditeur une licence non exclusive et
        strictement limitée aux besoins techniques nécessaires à l&apos;exécution du Service
        (traitement, transcription, traduction, synthèse). Aucun contenu n&apos;est utilisé à
        d&apos;autres fins que celles demandées par l&apos;utilisateur.
      </p>

      <h2>5. Limites et disponibilité</h2>
      <p>
        L&apos;éditeur s&apos;efforce d&apos;assurer la disponibilité du Service 24 h / 24 et 7 j / 7,
        sous réserve d&apos;opérations de maintenance planifiées ou d&apos;incidents indépendants de
        sa volonté. Aucun engagement de disponibilité contractuel n&apos;est pris dans le cadre des
        formules gratuites.
      </p>

      <h2>6. Modification des CGU</h2>
      <p>
        L&apos;éditeur peut modifier les présentes CGU à tout moment. Les utilisateurs sont informés
        des évolutions importantes par email ou notification dans l&apos;application. La poursuite
        de l&apos;utilisation du Service après notification vaut acceptation des nouvelles CGU.
      </p>

      <h2>7. Résiliation</h2>
      <p>
        L&apos;utilisateur peut supprimer son compte à tout moment depuis son espace personnel. La
        suppression entraîne l&apos;effacement définitif de ses données conformément à la
        <a href="/confidentialite"> politique de confidentialité</a>.
      </p>

      <h2>8. Droit applicable et juridiction</h2>
      <p>
        Les présentes CGU sont soumises au droit français. Tout litige relatif à leur interprétation
        ou à leur exécution relève, à défaut d&apos;accord amiable, de la compétence exclusive des
        tribunaux français.
      </p>
    </LegalLayout>
  );
}
