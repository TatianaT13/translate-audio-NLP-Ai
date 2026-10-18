import Script from "next/script";

/**
 * Analytics privacy-first via Plausible.
 *
 * Charge le script uniquement si NEXT_PUBLIC_PLAUSIBLE_DOMAIN est defini
 * (sinon no-op, utile en dev local). Script asynchrone, defere apres hydratation,
 * aucun cookie, aucune donnee personnelle collectee → RGPD-friendly, pas
 * de banniere de consentement necessaire.
 *
 * Pour envoyer un event custom cote client :
 *   import { trackEvent } from "@/lib/analytics";
 *   trackEvent("Signup", { tier: "pro" });
 */
export function Analytics() {
  const domain = process.env.NEXT_PUBLIC_PLAUSIBLE_DOMAIN;
  const src    = process.env.NEXT_PUBLIC_PLAUSIBLE_SRC || "https://plausible.io/js/script.js";

  if (!domain) return null;

  return (
    <Script
      defer
      data-domain={domain}
      src={src}
      strategy="afterInteractive"
    />
  );
}
