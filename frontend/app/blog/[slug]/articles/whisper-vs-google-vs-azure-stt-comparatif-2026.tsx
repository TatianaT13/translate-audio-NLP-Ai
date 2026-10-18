import Link from "next/link";

export function Comparatif() {
  return (
    <>
      <p>
        Trois moteurs de reconnaissance vocale reviennent dans la plupart des
        projets : <strong>Whisper</strong> (OpenAI),{" "}
        <strong>Google Speech-to-Text</strong> et <strong>Azure AI Speech</strong>{" "}
        (Microsoft). Tous trois donnent de bons résultats sur les grandes
        langues ; ils se distinguent surtout par leur mode de déploiement, leurs
        fonctions annexes et leur écosystème.
      </p>

      <h2>Vue d&apos;ensemble</h2>
      <table>
        <thead>
          <tr><th></th><th>Whisper</th><th>Google</th><th>Azure</th></tr>
        </thead>
        <tbody>
          <tr><td>Modèle ouvert, auto-hébergeable</td><td>Oui</td><td>Non</td><td>Non</td></tr>
          <tr><td>API managée</td><td>Oui (OpenAI)</td><td>Oui</td><td>Oui</td></tr>
          <tr><td>Diarisation intégrée</td><td>Non (outils tiers)</td><td>Oui</td><td>Oui</td></tr>
          <tr><td>Personnalisation du vocabulaire</td><td>Via le prompt</td><td>Oui</td><td>Oui (Custom Speech)</td></tr>
          <tr><td>Point fort</td><td>Qualité et flexibilité</td><td>Couverture des langues</td><td>Écosystème Microsoft</td></tr>
        </tbody>
      </table>
      <p>
        Les tarifs évoluent régulièrement : comparez-les sur les pages
        officielles de chaque fournisseur avant de choisir.
      </p>

      <h2>Whisper</h2>
      <p>
        Publié en open source par OpenAI, Whisper s&apos;est imposé comme une
        référence de qualité, notamment sur les langues européennes. Il est
        disponible via l&apos;API d&apos;OpenAI ou en auto-hébergement.
      </p>
      <ul>
        <li><strong>Pour</strong> : excellente qualité, modèle ouvert, possibilité de garder les données chez soi, prompt pour guider le vocabulaire.</li>
        <li><strong>Contre</strong> : pas de diarisation native ; peut produire du texte inventé sur les longs silences ou le bruit ; l&apos;auto-hébergement demande un GPU et du savoir-faire.</li>
      </ul>

      <h2>Google Speech-to-Text</h2>
      <p>
        Le service de Google se distingue par le nombre de langues et de
        variantes régionales couvertes, et par son intégration à Google Cloud.
      </p>
      <ul>
        <li><strong>Pour</strong> : très large couverture linguistique, diarisation intégrée, transcription en streaming.</li>
        <li><strong>Contre</strong> : service fermé, dépendance à Google Cloud.</li>
      </ul>

      <h2>Azure AI Speech</h2>
      <p>
        L&apos;offre de Microsoft est naturelle dans les organisations déjà
        équipées de Microsoft 365 et d&apos;Azure. Sa fonction Custom Speech
        permet d&apos;adapter le modèle à un vocabulaire métier (médical,
        juridique, industriel).
      </p>
      <ul>
        <li><strong>Pour</strong> : personnalisation poussée, choix de la région d&apos;hébergement, intégration à l&apos;écosystème Microsoft.</li>
        <li><strong>Contre</strong> : service fermé, documentation dispersée.</li>
      </ul>

      <h2>Comment choisir</h2>
      <ul>
        <li>
          <strong>Données sensibles ou gros volumes</strong> : Whisper
          auto-hébergé, pour garder la maîtrise des données et des coûts.
        </li>
        <li>
          <strong>Produit grand public sur les langues courantes</strong> :
          l&apos;API Whisper, complétée par un service de diarisation si besoin.
          C&apos;est le choix de <Link href="/">traduction-audio.fr</Link>{" "}
          (Whisper via OpenAI, AssemblyAI pour les intervenants).
        </li>
        <li>
          <strong>Langues rares ou variantes régionales</strong> : Google.
        </li>
        <li>
          <strong>Vocabulaire métier dans un environnement Microsoft</strong> :
          Azure avec Custom Speech.
        </li>
      </ul>
      <p>
        Dans tous les cas, testez sur <em>vos</em> enregistrements : la qualité
        varie fortement selon l&apos;accent, le micro et le domaine.
      </p>
    </>
  );
}
