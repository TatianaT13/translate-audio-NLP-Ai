import Link from "next/link";

export function Diarisation() {
  return (
    <>
      <p>
        La <strong>diarisation</strong> consiste à déterminer &laquo; qui parle,
        et quand &raquo; dans un enregistrement. Transcrire est une chose ;
        attribuer chaque phrase à son locuteur en est une autre, nettement plus
        difficile. Voici comment elle fonctionne, quand elle est utile et
        quelles précautions prendre.
      </p>

      <h2>Comment ça marche</h2>
      <ol>
        <li>
          <strong>Détection de la parole</strong> : l&apos;audio est découpé en
          segments où quelqu&apos;un parle, en écartant silences et bruits.
        </li>
        <li>
          <strong>Empreinte vocale</strong> : chaque segment est converti en un
          vecteur numérique qui résume les caractéristiques de la voix (timbre,
          hauteur, débit).
        </li>
        <li>
          <strong>Regroupement</strong> : les segments aux empreintes proches
          sont rassemblés. Chaque groupe correspond à un locuteur, nommé
          &laquo; A &raquo;, &laquo; B &raquo;, &laquo; C &raquo;…
        </li>
      </ol>
      <p>
        La principale difficulté vient des <strong>chevauchements</strong> :
        quand deux personnes parlent en même temps, le segment mélange deux
        voix et ne correspond proprement à aucun groupe.
      </p>

      <h2>Quand l&apos;utiliser</h2>
      <ul>
        <li><strong>Réunions</strong> : savoir qui a proposé, validé ou pris une action.</li>
        <li><strong>Interviews et podcasts</strong> : séparer animateur et invité pour le montage et les sous-titres.</li>
        <li><strong>Entretiens utilisateurs, focus groups</strong> : relier chaque opinion à son auteur.</li>
        <li><strong>Relation client</strong> : distinguer conseiller et client pour analyser les échanges.</li>
      </ul>

      <h2>Les outils</h2>
      <h3>Services en ligne (API)</h3>
      <p>
        Des services comme AssemblyAI proposent la diarisation par simple appel
        d&apos;API, sans avoir à indiquer le nombre de locuteurs à l&apos;avance.
        C&apos;est le moteur utilisé sur{" "}
        <Link href="/meeting">traduction-audio.fr</Link> pour l&apos;option
        &laquo; Identifier les intervenants &raquo;.
      </p>
      <h3>pyannote.audio</h3>
      <p>
        Bibliothèque open source de référence pour la diarisation,
        auto-hébergeable. Elle demande une infrastructure (idéalement un GPU)
        et des compétences en apprentissage automatique, mais garde les données
        chez vous.
      </p>
      <h3>WhisperX</h3>
      <p>
        Projet open source qui combine la transcription Whisper et la
        diarisation pyannote, avec un alignement au mot près. Puissant, mais
        plus lourd à mettre en place.
      </p>

      <h2>Les limites à connaître</h2>
      <ul>
        <li>
          <strong>Voix proches</strong> : deux voix au timbre similaire peuvent
          être fusionnées en un seul intervenant.
        </li>
        <li>
          <strong>Audio bruyant</strong> : dans une salle réverbérante ou avec
          des conversations croisées, la qualité baisse nettement.
        </li>
        <li>
          <strong>Étiquettes anonymes</strong> : &laquo; A &raquo; et
          &laquo; B &raquo; n&apos;ont pas de sens en soi. Associer une étiquette
          à une personne se fait à la main, ce que permet le renommage des
          intervenants sur traduction-audio.fr.
        </li>
      </ul>

      <h2>Et le RGPD ?</h2>
      <p>
        Une voix enregistrée est une donnée personnelle. Informez les personnes
        enregistrées et ne conservez les enregistrements que le temps
        nécessaire. La diarisation se contente de distinguer des voix ;
        l&apos;identification biométrique (reconnaître automatiquement une
        personne à sa voix) est beaucoup plus encadrée, et nous ne la proposons
        pas.
      </p>
    </>
  );
}
