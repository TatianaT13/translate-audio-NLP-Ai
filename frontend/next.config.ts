import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",

  // Reverse proxy intégré : derrière nginx (qui ne route que /), le frontend
  // Next.js proxie toutes les requêtes /api, /pipeline, /stt, /llm vers les
  // services backend du réseau Docker interne. Évite d'avoir à configurer
  // de multiples location nginx côté serveur.
  async rewrites() {
    return [
      { source: "/api/:path*",      destination: "http://gateway:8004/:path*" },
      { source: "/pipeline/:path*", destination: "http://pipeline:8000/:path*" },
      { source: "/stt/:path*",      destination: "http://stt:8001/:path*"     },
      { source: "/llm/:path*",      destination: "http://llm:8002/:path*"     },
    ];
  },
};

export default nextConfig;
