import Link from "next/link";

export function CompteRenduReunion() {
  return (
    <>
      <p>
        Rédiger un compte-rendu de réunion prend souvent autant de temps que la
        réunion elle-même : réécouter, retrouver qui a dit quoi, extraire les
        décisions et les actions. Les outils de transcription par IA permettent
        aujourd&apos;hui d&apos;automatiser l&apos;essentiel de ce travail, à
        condition de comprendre ce qu&apos;ils font bien et ce qu&apos;il faut
        encore vérifier.
      </p>

      <h2>Les trois briques d&apos;un compte-rendu automatique</h2>
      <h3>1. La transcription</h3>
      <p>
        Un modèle de reconnaissance vocale (comme Whisper) convertit la parole
        en texte. Sur un enregistrement propre, la qualité est aujourd&apos;hui
        très bonne ; les erreurs se concentrent sur les noms propres, les
        acronymes et le jargon métier.
      </p>
      <h3>2. L&apos;identification des intervenants</h3>
      <p>
        Une transcription brute ne dit pas qui parle. La <strong>diarisation</strong>{" "}
        découpe l&apos;audio par locuteur (&laquo; Intervenant A &raquo;,
        &laquo; Intervenant B &raquo;…). C&apos;est ce qui rend un compte-rendu
        actionnable : savoir qui a proposé, qui a validé, qui prend
        l&apos;action. Nous détaillons son fonctionnement dans{" "}
        <Link href="/blog/identifier-intervenants-audio-diarisation">
          cet article sur la diarisation
        </Link>.
      </p>
      <h3>3. La synthèse</h3>
      <p>
        Un modèle de langage transforme le transcript en compte-rendu structuré :
        points abordés, décisions, actions et responsables. Si le transcript
        contient les noms des intervenants, la synthèse peut attribuer chaque
        décision à la bonne personne.
      </p>

      <h2>La méthode avec traduction-audio.fr</h2>
      <p>
        La page <Link href="/meeting">Compte-rendu</Link> enregistre la réunion
        via le micro de votre ordinateur. Elle convient donc aux réunions en
        présentiel, ou aux visioconférences dont le son sort par les haut-parleurs.
      </p>
      <ol>
        <li>
          Activez <strong>&laquo; Identifier les intervenants &raquo;</strong>{" "}
          avant de lancer l&apos;enregistrement.
        </li>
        <li>
          À la fin de la réunion, arrêtez l&apos;enregistrement puis lancez
          l&apos;identification : le transcript s&apos;affiche par intervenant,
          avec les horodatages.
        </li>
        <li>
          <strong>Renommez les intervenants</strong> : &laquo; Intervenant A &raquo;
          devient &laquo; Alice Martin &raquo;. L&apos;IA distingue les voix mais
          ne sait pas à qui elles appartiennent ; c&apos;est à vous de faire le
          lien.
        </li>
        <li>
          Choisissez la langue et le style du compte-rendu, puis générez-le.
          Exportez-le en Markdown ou en PDF, et le transcript en sous-titres SRT.
        </li>
      </ol>
      <p>
        Pour un fichier déjà enregistré (export de visioconférence, dictaphone),
        utilisez plutôt la page <Link href="/translate">Traduction</Link> avec
        l&apos;identification des intervenants : vous obtenez la transcription et
        sa traduction par intervenant (fichiers jusqu&apos;à 25 Mo).
      </p>

      <h2>Ce qu&apos;il faut toujours vérifier</h2>
      <h3>Les noms propres et les chiffres</h3>
      <p>
        Un nom mal transcrit ou un montant approximatif peut se propager dans
        la synthèse. Relisez ces éléments avant de diffuser le compte-rendu.
      </p>
      <h3>Les voix qui se chevauchent</h3>
      <p>
        Quand plusieurs personnes parlent en même temps, ou dans une salle
        bruyante, l&apos;attribution par intervenant devient moins fiable. Un
        micro de conférence placé au centre de la table améliore nettement le
        résultat.
      </p>
      <h3>Le consentement des participants</h3>
      <p>
        Informez les participants en début de réunion qu&apos;elle est
        enregistrée et transcrite par un outil d&apos;IA. Pour savoir comment
        l&apos;audio est traité et par quels prestataires, consultez notre{" "}
        <Link href="/confidentialite">politique de confidentialité</Link>.
      </p>
    </>
  );
}
