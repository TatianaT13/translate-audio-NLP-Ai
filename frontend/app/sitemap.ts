import type { MetadataRoute } from "next";
import { ARTICLES } from "@/lib/blog";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://traduction-audio.fr";

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  const routes: { path: string; priority: number; changeFrequency: MetadataRoute.Sitemap[number]["changeFrequency"]; lastModified?: Date }[] = [
    // /translate, /meeting, /live : exclus (redirection login + disallow robots.txt)
    { path: "",                   priority: 1.0, changeFrequency: "weekly"  },
    { path: "/tarifs",            priority: 0.9, changeFrequency: "weekly"  },
    { path: "/blog",              priority: 0.8, changeFrequency: "weekly"  },
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

  const staticEntries = routes.map(r => ({
    url:            `${SITE_URL}${r.path}`,
    lastModified:   r.lastModified || now,
    changeFrequency: r.changeFrequency,
    priority:        r.priority,
  }));

  const articleEntries = ARTICLES.map(a => ({
    url:            `${SITE_URL}/blog/${a.slug}`,
    lastModified:   new Date(a.date),
    changeFrequency: "monthly" as const,
    priority:        0.7,
  }));

  return [...staticEntries, ...articleEntries];
}
