import { LegalLayout } from "@/components/LegalLayout";

export const metadata = { title: "Politique de confidentialité — traduction-audio.fr" };

export default function Confidentialite() {
  return (
    <LegalLayout title="Politique de confidentialité" updated="15 septembre 2026">
      <p>
        La présente politique décrit la manière dont <strong>traduction-audio.fr</strong> collecte,
        utilise et protège les données personnelles de ses utilisateurs, conformément au
        <strong> Règlement général sur la protection des données (RGPD)</strong> et à la loi
        « Informatique et Libertés ».
      </p>

      <h2>1. Responsable du traitement</h2>
      <p>
        <strong>Tetyana Tarasenko</strong>, entrepreneur individuel (SIRET 838 177 012 00021), dont
        les coordonnées figurent dans les <a href="/mentions-legales">mentions légales</a>, est
        responsable du traitement de vos données.
      </p>

      <h2>2. Données collectées</h2>
      <h3>Données de compte</h3>
      <ul>
        <li>Adresse email (identifiant du compte)</li>
        <li>Mot de passe (stocké sous forme chiffrée irréversible — <em>bcrypt</em>)</li>
        <li>Date de création du compte et dernière connexion</li>
      </ul>
      <h3>Données d&apos;usage</h3>
      <ul>
        <li>Fichiers audio téléversés — traités puis <strong>supprimés à la fin du traitement</strong>, non conservés</li>
        <li>Textes transcrits et traductions générées — conservés dans votre espace personnel jusqu&apos;à suppression manuelle</li>
        <li>Métriques techniques anonymisées (durée de traitement, langue cible, taille du fichier)</li>
      </ul>
      <h3>Données de facturation</h3>
      <ul>
        <li>Historique des paiements et factures (obligation légale : conservation 10 ans)</li>
        <li>Les données bancaires sont traitées exclusivement par notre prestataire de paiement (Stripe Payments Europe, Ltd.) et ne transitent jamais par nos serveurs</li>
      </ul>

      <h2>3. Finalités du traitement</h2>
      <ul>
        <li>Fournir et faire fonctionner le Service (transcription, traduction, synthèse)</li>
        <li>Gérer votre compte et l&apos;authentification</li>
        <li>Assurer la facturation et le suivi comptable</li>
        <li>Améliorer la qualité du Service via des statistiques anonymisées</li>
        <li>Répondre à vos demandes de support</li>
        <li>Respecter nos obligations légales</li>
      </ul>

      <h2>4. Base légale du traitement</h2>
      <ul>
        <li><strong>Exécution du contrat</strong> — pour les données nécessaires à la fourniture du Service</li>
        <li><strong>Obligation légale</strong> — pour les données de facturation et comptables</li>
        <li><strong>Intérêt légitime</strong> — pour les métriques techniques anonymisées</li>
        <li><strong>Consentement</strong> — pour les cookies non essentiels et communications marketing</li>
      </ul>

      <h2>5. Durée de conservation</h2>
      <ul>
        <li>Compte utilisateur : jusqu&apos;à suppression par l&apos;utilisateur ou après 3 ans d&apos;inactivité</li>
        <li>Fichiers audio : supprimés à la fin du traitement (aucune conservation)</li>
        <li>Transcriptions et traductions : jusqu&apos;à suppression manuelle depuis votre espace</li>
        <li>Factures : 10 ans (obligation légale)</li>
        <li>Logs techniques : 12 mois maximum</li>
      </ul>

      <h2>6. Destinataires des données</h2>
      <p>
        Vos données ne sont ni vendues, ni louées, ni transmises à des tiers à des fins commerciales.
        Elles peuvent être partagées avec :
      </p>
      <ul>
        <li>Nos sous-traitants techniques (hébergeur, prestataire de paiement, fournisseurs de modèles d&apos;IA), strictement pour l&apos;exécution du Service</li>
        <li>Les autorités compétentes, sur réquisition légale</li>
      </ul>

      <h2>7. Transferts hors Union européenne</h2>
      <p>
        Certains sous-traitants (fournisseurs de modèles d&apos;IA) peuvent traiter vos données hors
        de l&apos;Union européenne, notamment aux États-Unis. Ces transferts sont encadrés par des
        clauses contractuelles types approuvées par la Commission européenne ou par des mécanismes
        équivalents.
      </p>

      <h2>8. Vos droits</h2>
      <p>Conformément au RGPD, vous disposez des droits suivants :</p>
      <ul>
        <li><strong>Accès</strong> : obtenir une copie de vos données</li>
        <li><strong>Rectification</strong> : corriger vos données inexactes</li>
        <li><strong>Effacement</strong> : demander la suppression de vos données</li>
        <li><strong>Opposition</strong> : vous opposer à certains traitements</li>
        <li><strong>Limitation</strong> : restreindre temporairement le traitement</li>
        <li><strong>Portabilité</strong> : recevoir vos données dans un format structuré</li>
        <li><strong>Retrait du consentement</strong> à tout moment</li>
      </ul>
      <p>
        Pour exercer ces droits, contactez-nous via <a href="/contact">notre formulaire</a>. Vous
        pouvez également introduire une réclamation auprès de la <strong>CNIL</strong> (
        <a href="https://www.cnil.fr" target="_blank" rel="noopener noreferrer">www.cnil.fr</a>).
      </p>

      <h2>9. Sécurité</h2>
      <p>
        Nous mettons en œuvre des mesures techniques et organisationnelles adaptées pour protéger
        vos données : chiffrement HTTPS, mots de passe chiffrés (bcrypt), tokens d&apos;authentification
        temporaires, hébergement dans l&apos;Union européenne, contrôle d&apos;accès strict aux
        systèmes.
      </p>

      <h2>10. Contact</h2>
      <p>
        Pour toute question relative à cette politique ou à vos données personnelles, écrivez-nous
        via <a href="/contact">notre formulaire de contact</a>.
      </p>
    </LegalLayout>
  );
}
