import Link from "next/link";

export function YoutubeSousTitres() {
  return (
    <>
      <p>
        YouTube génère des sous-titres automatiques et propose au spectateur
        une traduction automatique à la volée, mais vous ne contrôlez ni la
        qualité ni le vocabulaire. Pour une audience internationale, fournir vos
        propres sous-titres traduits, au format SRT, reste la méthode la plus
        fiable.
      </p>

      <h2>Le format SRT</h2>
      <p>Un fichier SRT est un simple fichier texte, structuré ainsi :</p>
      <blockquote>
        1<br />
        00:00:00,000 --&gt; 00:00:03,500<br />
        Bienvenue dans ce tutoriel.<br />
        <br />
        2<br />
        00:00:03,800 --&gt; 00:00:06,200<br />
        Aujourd&apos;hui, on parle d&apos;IA.
      </blockquote>
      <p>
        YouTube Studio accepte un fichier SRT par langue : vous pouvez en
        ajouter autant que de langues cibles.
      </p>

      <h2>La méthode en quatre étapes</h2>
      <h3>1. Récupérer l&apos;audio</h3>
      <p>
        Partez de votre fichier vidéo source, ou téléchargez votre vidéo depuis
        YouTube Studio. Extrayez ensuite la piste audio, par exemple avec FFmpeg :
      </p>
      <blockquote>
        <code>ffmpeg -i video.mp4 -vn -c:a libmp3lame -b:a 96k audio.mp3</code>
      </blockquote>
      <p>Un débit de 96 kb/s suffit pour la voix et réduit la taille du fichier.</p>

      <h3>2. Transcrire et traduire</h3>
      <p>
        Sur <Link href="/translate">traduction-audio.fr</Link>, importez le MP3
        (25 Mo maximum) et activez{" "}
        <strong>&laquo; Identifier les intervenants &raquo;</strong> : c&apos;est
        cette option qui produit les segments horodatés nécessaires au SRT. Les
        langues cibles disponibles sont l&apos;anglais, l&apos;espagnol,
        l&apos;allemand et l&apos;ukrainien, à partir d&apos;un audio en français.
      </p>

      <h3>3. Exporter et relire le SRT</h3>
      <p>
        Le bouton <code>.srt</code> télécharge les sous-titres traduits. Chaque
        segment est préfixé par le nom de l&apos;intervenant ; supprimez ces
        préfixes si vous n&apos;en voulez pas à l&apos;écran. Les segments suivent
        les prises de parole et peuvent être longs : découpez-les si besoin avec
        un éditeur de sous-titres gratuit comme Aegisub ou Subtitle Edit.
      </p>

      <h3>4. Importer dans YouTube Studio</h3>
      <ol>
        <li>Ouvrez la vidéo dans YouTube Studio, rubrique Sous-titres.</li>
        <li>Ajoutez la langue cible.</li>
        <li>Importez le fichier SRT correspondant, en indiquant qu&apos;il contient des horodatages.</li>
        <li>Vérifiez la synchronisation sur quelques passages, puis publiez.</li>
      </ol>

      <h2>Règles de lisibilité</h2>
      <ul>
        <li>
          <strong>Lignes courtes</strong> : environ 42 caractères par ligne et
          deux lignes maximum, une règle courante chez les professionnels du
          sous-titrage.
        </li>
        <li>
          <strong>Durée d&apos;affichage</strong> : évitez qu&apos;un même
          sous-titre reste plus de 6 à 7 secondes à l&apos;écran ; au-delà,
          mieux vaut le scinder.
        </li>
        <li>
          <strong>Relecture du début</strong> : les erreurs se concentrent
          souvent dans les premières secondes (musique d&apos;introduction, bruit
          de fond). Relisez-les en priorité.
        </li>
      </ul>
    </>
  );
}
