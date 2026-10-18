import type { MetadataRoute } from "next";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://traduction-audio.fr";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        // Le favicon est servi sous /api : il doit rester crawlable pour Google.
        allow:    ["/", "/api/favicon.ico"],
        // Routes authentifiees ou administratives : non indexables.
        disallow: ["/admin", "/api", "/live", "/translate", "/meeting",
                   "/reset-password", "/forgot-password"],
      },
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
    host:    SITE_URL,
  };
}
