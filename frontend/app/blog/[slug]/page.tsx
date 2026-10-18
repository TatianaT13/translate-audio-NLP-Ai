import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { BlogLayout } from "@/components/BlogLayout";
import { ARTICLES, getArticleBySlug } from "@/lib/blog";
import { ArticleContent } from "./articles";

export async function generateStaticParams() {
  return ARTICLES.map(a => ({ slug: a.slug }));
}

type Params = { slug: string };

export async function generateMetadata(
  { params }: { params: Promise<Params> },
): Promise<Metadata> {
  const { slug } = await params;
  const article = getArticleBySlug(slug);
  if (!article) return { title: "Article introuvable" };
  return {
    title:       article.title,
    description: article.description,
    keywords:    article.keywords,
    alternates:  { canonical: `/blog/${article.slug}` },
    openGraph: {
      type:        "article",
      title:       article.title,
      description: article.description,
      url:         `/blog/${article.slug}`,
      publishedTime: article.date,
    },
    twitter: {
      card:        "summary_large_image",
      title:       article.title,
      description: article.description,
    },
  };
}

export default async function BlogArticlePage(
  { params }: { params: Promise<Params> },
) {
  const { slug } = await params;
  const article = getArticleBySlug(slug);
  if (!article) notFound();
  return (
    <BlogLayout article={article}>
      <ArticleContent slug={slug} />
    </BlogLayout>
  );
}
