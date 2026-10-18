// Contenu de chaque article : frontend/app/blog/[slug]/articles/<slug>.tsx

export interface BlogArticle {
  slug:        string;
  title:       string;
  description: string;      // meta description (< 160 caractères)
  excerpt:     string;
  date:        string;      // YYYY-MM-DD
  readMinutes: number;
  keywords:    string[];
  category:    "guide" | "comparatif" | "tutoriel";
}

export const ARTICLES: BlogArticle[] = [
  {
    slug: "compte-rendu-reunion-automatique",
    title: "Compte-rendu de réunion automatique : transcrire et identifier les intervenants",
    description: "Comment obtenir un compte-rendu de réunion structuré avec l'IA : enregistrement, identification des intervenants, résumé, export PDF. Méthode et limites.",
    excerpt: "Réunion d'équipe, comité, entretien : comment passer de l'enregistrement à un compte-rendu attribué par intervenant, sans tout recopier.",
    date: "2026-10-18",
    readMinutes: 6,
    keywords: ["compte-rendu de réunion", "transcription réunion", "identification des intervenants", "diarisation", "résumé automatique"],
    category: "guide",
  },
  {
    slug: "traduire-podcast-francais-anglais-ia",
    title: "Traduire un podcast français en anglais avec l'IA : le workflow",
    description: "Transcription, traduction, voix de synthèse ou voix clonée, sous-titres : les étapes pour publier la version anglaise d'un podcast français, et les pièges à éviter.",
    excerpt: "Vous publiez un podcast en français et visez une audience anglophone : voici les étapes, les choix de voix et les pièges juridiques.",
    date: "2026-10-18",
    readMinutes: 7,
    keywords: ["traduire podcast", "podcast anglais", "voix de synthèse", "voix clonée", "sous-titres SRT"],
    category: "tutoriel",
  },
  {
    slug: "identifier-intervenants-audio-diarisation",
    title: "Identifier qui parle dans un enregistrement : la diarisation expliquée",
    description: "Diarisation audio : comment l'IA attribue chaque phrase à son locuteur, dans quels cas l'utiliser, quels outils existent et quelles précautions RGPD prendre.",
    excerpt: "Interviews, focus groups, réunions : la diarisation attribue chaque phrase à son locuteur. Fonctionnement, outils et limites.",
    date: "2026-10-18",
    readMinutes: 6,
    keywords: ["diarisation", "identification du locuteur", "transcription multi-locuteurs", "assemblyai", "pyannote"],
    category: "guide",
  },
  {
    slug: "whisper-vs-google-vs-azure-stt-comparatif-2026",
    title: "Whisper, Google Speech-to-Text ou Azure Speech : lequel choisir ?",
    description: "Comparatif des trois grands moteurs de transcription audio : points forts, limites, modèles de déploiement et cas d'usage où chacun s'impose.",
    excerpt: "Vous construisez un produit de transcription ou vous évaluez un outil : quel moteur de reconnaissance vocale choisir selon votre usage ?",
    date: "2026-10-18",
    readMinutes: 6,
    keywords: ["whisper", "google speech-to-text", "azure speech", "comparatif transcription", "reconnaissance vocale"],
    category: "comparatif",
  },
  {
    slug: "sous-titrer-video-youtube-automatiquement",
    title: "Sous-titrer et traduire une vidéo YouTube avec des fichiers SRT",
    description: "Générer des sous-titres SRT traduits pour YouTube : extraction de l'audio, transcription, traduction, import dans YouTube Studio et règles de lisibilité.",
    excerpt: "La traduction automatique du lecteur YouTube ne vous laisse aucun contrôle. Voici comment fournir vos propres sous-titres traduits.",
    date: "2026-10-18",
    readMinutes: 5,
    keywords: ["sous-titres youtube", "fichier srt", "traduire sous-titres", "youtube studio", "sous-titrage automatique"],
    category: "tutoriel",
  },
];

export function getArticleBySlug(slug: string): BlogArticle | undefined {
  return ARTICLES.find(a => a.slug === slug);
}
