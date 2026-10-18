import { CompteRenduReunion } from "./compte-rendu-reunion-automatique";
import { PodcastFrAnglais } from "./traduire-podcast-francais-anglais-ia";
import { Diarisation } from "./identifier-intervenants-audio-diarisation";
import { Comparatif } from "./whisper-vs-google-vs-azure-stt-comparatif-2026";
import { YoutubeSousTitres } from "./sous-titrer-video-youtube-automatiquement";

const CONTENT: Record<string, React.FC> = {
  "compte-rendu-reunion-automatique":          CompteRenduReunion,
  "traduire-podcast-francais-anglais-ia":          PodcastFrAnglais,
  "identifier-intervenants-audio-diarisation":     Diarisation,
  "whisper-vs-google-vs-azure-stt-comparatif-2026": Comparatif,
  "sous-titrer-video-youtube-automatiquement":     YoutubeSousTitres,
};

export function ArticleContent({ slug }: { slug: string }) {
  const C = CONTENT[slug];
  if (!C) return <p>Contenu introuvable.</p>;
  return <C />;
}
