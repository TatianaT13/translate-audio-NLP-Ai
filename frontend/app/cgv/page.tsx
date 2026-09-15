import { LegalLayout } from "@/components/LegalLayout";

export const metadata = { title: "Conditions générales de vente — traduction-audio.fr" };

export default function CGV() {
  return (
    <LegalLayout title="Conditions générales de vente" updated="15 septembre 2026">
      <p>
        Les présentes Conditions générales de vente (« <strong>CGV</strong> ») régissent la
        souscription aux offres payantes proposées par <strong>Tetyana Tarasenko</strong>,
        entrepreneur individuel (SIRET 838 177 012 00021), via le site <strong>traduction-audio.fr</strong>.
      </p>

      <h2>1. Objet</h2>
      <p>
        Les présentes CGV définissent les droits et obligations des parties dans le cadre de la
        commercialisation des formules d&apos;abonnement et des services à l&apos;usage du Service.
      </p>

      <h2>2. Offres et tarifs</h2>
      <p>
        Les offres, leurs caractéristiques et leurs tarifs sont détaillés sur la page dédiée aux
        tarifs du site. Les prix sont exprimés en euros. En application de l&apos;article 293 B du
        Code général des impôts, l&apos;éditeur bénéficie de la franchise en base de TVA :
        <em> TVA non applicable, art. 293 B du CGI</em>. L&apos;éditeur se réserve le droit de
        modifier ses tarifs à tout moment, sans effet rétroactif sur les abonnements en cours.
      </p>

      <h2>3. Commande</h2>
      <p>
        Toute commande implique l&apos;acceptation pleine et entière des présentes CGV. La commande
        est validée après confirmation du paiement par le prestataire de services de paiement retenu
        (Stripe Payments Europe, Ltd.).
      </p>

      <h2>4. Modalités de paiement</h2>
      <p>
        Le paiement s&apos;effectue en ligne, par carte bancaire ou tout autre moyen proposé sur la
        page de paiement. Les transactions sont sécurisées par le prestataire de paiement, qui traite
        les données bancaires de l&apos;utilisateur. L&apos;éditeur n&apos;a accès à aucune donnée
        bancaire.
      </p>

      <h2>5. Facturation</h2>
      <p>
        Les factures sont émises automatiquement à chaque échéance et mises à disposition de
        l&apos;utilisateur depuis son espace personnel.
      </p>

      <h2>6. Droit de rétractation</h2>
      <p>
        Conformément à l&apos;article <strong>L. 221-28 du Code de la consommation</strong>, le droit
        de rétractation ne peut être exercé pour les contrats de fourniture de contenu numérique non
        fourni sur un support matériel dont l&apos;exécution a commencé après accord préalable exprès
        du consommateur et renoncement exprès à son droit de rétractation.
      </p>
      <p>
        Lors de la souscription, l&apos;utilisateur consent expressément à l&apos;exécution immédiate
        du Service et renonce à son droit de rétractation.
      </p>

      <h2>7. Durée et résiliation</h2>
      <p>
        Les abonnements sont conclus pour la durée choisie (mensuelle, annuelle). Ils se renouvellent
        tacitement à échéance, sauf résiliation par l&apos;utilisateur avant la date de renouvellement
        depuis son espace personnel. La résiliation prend effet à la fin de la période en cours ;
        aucun remboursement au prorata n&apos;est effectué.
      </p>

      <h2>8. Suspension et défaut de paiement</h2>
      <p>
        En cas de défaut de paiement, l&apos;éditeur se réserve le droit de suspendre l&apos;accès
        au Service jusqu&apos;à régularisation. Après un délai de 15 jours sans régularisation, le
        compte pourra être supprimé.
      </p>

      <h2>9. Responsabilité</h2>
      <p>
        L&apos;éditeur s&apos;engage à mettre en œuvre les moyens nécessaires au bon fonctionnement
        du Service. Il ne saurait être tenu responsable des dommages indirects, notamment des pertes
        de données, de chiffre d&apos;affaires ou d&apos;image, résultant de l&apos;utilisation ou de
        l&apos;impossibilité d&apos;utilisation du Service.
      </p>
      <p>
        Le Service repose sur des modèles d&apos;intelligence artificielle. Les traductions et
        transcriptions générées ne sauraient être garanties comme fidèles au mot près et ne peuvent
        remplacer une traduction professionnelle certifiée dans les contextes qui l&apos;exigent
        (juridique, médical, administratif).
      </p>

      <h2>10. Médiation de la consommation</h2>
      <p>
        Conformément aux articles L. 611-1 et suivants du Code de la consommation, tout consommateur
        a le droit de recourir gratuitement à un médiateur de la consommation en vue de la résolution
        amiable de tout litige. En cas d&apos;échec d&apos;une réclamation formulée par écrit auprès
        de l&apos;éditeur, le consommateur peut également saisir la plateforme européenne de
        règlement en ligne des litiges :
        {" "}
        <a href="https://ec.europa.eu/consumers/odr" target="_blank" rel="noopener noreferrer">
          ec.europa.eu/consumers/odr
        </a>.
      </p>
      <p style={{ fontSize: "12px", opacity: 0.7 }}>
        <em>
          Note : l&apos;adhésion à un service de médiation de la consommation agréé sera indiquée
          ici dès qu&apos;elle sera effective.
        </em>
      </p>

      <h2>11. Droit applicable et juridiction</h2>
      <p>
        Les présentes CGV sont soumises au droit français. Tout litige relatif à leur interprétation
        ou à leur exécution relève de la compétence exclusive des tribunaux français, sauf disposition
        d&apos;ordre public plus favorable au consommateur.
      </p>
    </LegalLayout>
  );
}
