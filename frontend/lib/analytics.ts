/**
 * Helpers Plausible — custom events + goals.
 *
 * Plausible expose window.plausible(eventName, { props }) une fois le script
 * charge. Tant qu'il n'est pas pret, les events sont bufferises par la lib
 * Plausible elle-meme (via le script-outbound-only sinon on wrap avec queue).
 */

declare global {
  interface Window {
    plausible?: (event: string, options?: { props?: Record<string, string | number | boolean>; callback?: () => void }) => void;
  }
}

/** Envoie un event custom Plausible. No-op si Plausible n'est pas charge. */
export function trackEvent(
  name: string,
  props?: Record<string, string | number | boolean>,
): void {
  if (typeof window === "undefined" || !window.plausible) return;
  try {
    window.plausible(name, props ? { props } : undefined);
  } catch (e) {
    console.warn("[analytics] trackEvent error:", e);
  }
}

/** Goals metier — a appeler aux moments cles de conversion. */
export const Goals = {
  signup:        () => trackEvent("Signup"),
  login:         () => trackEvent("Login"),
  translateUsed: (lang: string)  => trackEvent("TranslateUsed", { lang }),
  meetingUsed:   (diarized: boolean) => trackEvent("MeetingUsed", { diarized }),
  liveUsed:      () => trackEvent("LiveUsed"),
  checkoutStart: (tier: string)  => trackEvent("CheckoutStart", { tier }),
  checkoutDone:  (tier: string)  => trackEvent("CheckoutDone", { tier }),
};
