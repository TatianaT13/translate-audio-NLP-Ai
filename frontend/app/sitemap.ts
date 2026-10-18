import type { MetadataRoute } from "next";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://traduction-audio.fr";

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  const routes: { path: string; priority: number; changeFrequency: MetadataRoute.Sitemap[number]["changeFrequency"] }[] = [
    { path: "",                   priority: 1.0, changeFrequency: "weekly"  },
    { path: "/translate",         priority: 0.9, changeFrequency: "monthly" },
    { path: "/meeting",           priority: 0.9, changeFrequency: "monthly" },
    { path: "/live",              priority: 0.8, changeFrequency: "monthly" },
    { path: "/tarifs",            priority: 0.9, changeFrequency: "weekly"  },
    { path: "/a-propos",          priority: 0.6, changeFrequency: "monthly" },
    { path: "/contact",           priority: 0.6, changeFrequency: "monthly" },
    { path: "/login",             priority: 0.4, changeFrequency: "yearly"  },
    { path: "/register",          priority: 0.5, changeFrequency: "yearly"  },
    { path: "/cgu",               priority: 0.3, changeFrequency: "yearly"  },
    { path: "/cgv",               priority: 0.3, changeFrequency: "yearly"  },
    { path: "/confidentialite",   priority: 0.3, changeFrequency: "yearly"  },
    { path: "/mentions-legales",  priority: 0.3, changeFrequency: "yearly"  },
    { path: "/cookies",           priority: 0.3, changeFrequency: "yearly"  },
  ];
  return routes.map(r => ({
    url:            `${SITE_URL}${r.path}`,
    lastModified:   now,
    changeFrequency: r.changeFrequency,
    priority:        r.priority,
  }));
}
