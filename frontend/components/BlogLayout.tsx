import { AppHeader } from "@/components/AppHeader";
import { FooterFull } from "@/components/Footer";
import Link from "next/link";
import type { BlogArticle } from "@/lib/blog";

interface BlogLayoutProps {
  article:  BlogArticle;
  children: React.ReactNode;
}

export function BlogLayout({ article, children }: BlogLayoutProps) {
  const dateFr = new Date(article.date).toLocaleDateString("fr-FR", {
    year: "numeric", month: "long", day: "numeric",
  });
  return (
    <>
      <AppHeader variant="landing" />
      <main style={{ maxWidth: "720px", margin: "0 auto", padding: "48px 24px 64px" }}>
        <nav style={{ fontSize: "12px", marginBottom: "20px" }}>
          <Link href="/blog" style={{ color: "var(--muted)", textDecoration: "none" }}>
            ← Tous les articles
          </Link>
        </nav>

        <header style={{ marginBottom: "32px" }}>
          <p style={{
            fontSize: "10px", letterSpacing: "0.25em", textTransform: "uppercase",
            color: "var(--accent)", margin: "0 0 12px", fontWeight: 500,
          }}>
            {article.category.toUpperCase()} · {article.readMinutes} min de lecture
          </p>
          <h1 style={{
            fontFamily: "var(--font-display), serif",
            fontSize: "clamp(1.75rem, 4vw, 2.5rem)",
            fontWeight: 400, letterSpacing: "-0.015em",
            color: "var(--foreground)", margin: "0 0 16px",
            lineHeight: 1.2,
          }}>
            {article.title}
          </h1>
          <p style={{ fontSize: "12px", color: "var(--muted)", fontFamily: "ui-monospace, monospace", margin: 0 }}>
            Publié le {dateFr}
          </p>
        </header>

        <article style={{ color: "var(--foreground)", fontSize: "16px", lineHeight: 1.75 }}>
          {children}
        </article>

        <aside style={{
          marginTop: "48px", padding: "24px",
          background: "rgba(201,169,110,0.04)",
          border: "1px solid var(--accent-dim)",
          borderRadius: "16px",
        }}>
          <p style={{ fontSize: "11px", letterSpacing: "0.2em", textTransform: "uppercase", color: "var(--accent)", margin: "0 0 12px", fontWeight: 600 }}>
            Essayer traduction-audio.fr
          </p>
          <p style={{ margin: "0 0 16px", color: "var(--muted)" }}>
            Transcrivez vos audios en français, traduisez-les en anglais, espagnol,
            allemand ou ukrainien, et générez vos comptes-rendus de réunion. 10 crédits
            offerts à l&apos;inscription, sans carte bancaire.
          </p>
          <Link
            href="/register"
            style={{
              display: "inline-block",
              background: "var(--accent)", color: "var(--background)",
              padding: "10px 22px", borderRadius: "8px",
              textDecoration: "none", fontWeight: 500, fontSize: "14px",
            }}
          >
            Créer un compte gratuit
          </Link>
        </aside>
      </main>
      <FooterFull />
      <style>{`
        article h2 { font-family: var(--font-display), serif; font-size: 1.5rem; font-weight: 400; color: var(--foreground); margin: 40px 0 16px; }
        article h3 { font-size: 1.1rem; font-weight: 600; color: var(--foreground); margin: 28px 0 10px; }
        article p { margin: 0 0 18px; color: var(--muted); }
        article strong { color: var(--foreground); font-weight: 500; }
        article ul, article ol { margin: 0 0 18px; padding-left: 24px; color: var(--muted); }
        article li { margin-bottom: 8px; }
        article a { color: var(--accent); text-decoration: underline; text-underline-offset: 2px; }
        article code { font-family: ui-monospace, monospace; background: rgba(201,169,110,0.08); padding: 2px 6px; border-radius: 4px; font-size: 0.9em; color: var(--accent); }
        article blockquote { margin: 24px 0; padding: 12px 20px; border-left: 3px solid var(--accent-dim); color: var(--foreground); font-style: italic; background: rgba(201,169,110,0.03); }
        article table { width: 100%; border-collapse: collapse; margin: 24px 0; font-size: 14px; }
        article th, article td { padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; color: var(--muted); }
        article th { color: var(--foreground); font-weight: 600; }
      `}</style>
    </>
  );
}
