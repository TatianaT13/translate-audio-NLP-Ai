import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",

  // Reverse proxy intégré : derrière nginx (qui ne route que /), le frontend
  // Next.js proxie toutes les requêtes /api, /pipeline, /stt, /llm vers les
  // services backend du réseau Docker interne. Évite d'avoir à configurer
  // de multiples location nginx côté serveur.
  async rewrites() {
    return {
      // beforeFiles s'applique AVANT le static serving et AVANT afterFiles :
      // /api/favicon.ico doit être intercepté ici pour ne pas partir vers la Gateway.
      // /api/* est exempté de basic auth nginx → parfait pour servir le favicon
      // sans qu'il soit bloqué par le login avant que l'user ne soit authentifié.
      beforeFiles: [
        { source: "/api/favicon.ico", destination: "/favicon.ico" },
      ],
      afterFiles: [
        { source: "/api/:path*",      destination: "http://gateway:8004/:path*" },
        { source: "/pipeline/:path*", destination: "http://pipeline:8000/:path*" },
        { source: "/stt/:path*",      destination: "http://stt:8001/:path*"     },
        { source: "/llm/:path*",      destination: "http://llm:8002/:path*"     },
      ],
      fallback: [],
    };
  },

  // Le pipeline STT + LLM + TTS peut prendre plus d'une minute sur un audio long.
  // Par défaut, Next.js coupe les rewrites à ~30s → ECONNRESET côté user.
  experimental: {
    proxyTimeout: 600_000, // 10 minutes en ms
  },
};

export default nextConfig;
