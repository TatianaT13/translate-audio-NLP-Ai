"use client";

import { useEffect } from "react";

const DSN = process.env.NEXT_PUBLIC_SENTRY_DSN;

/** Sentry navigateur, charge uniquement si un DSN est configure (aucun poids sinon). */
export function ErrorMonitoring() {
  useEffect(() => {
    if (!DSN) return;
    import("@sentry/browser").then((Sentry) => {
      Sentry.init({
        dsn: DSN,
        sendDefaultPii: false,
        tracesSampleRate: 0,
        // Les fils d'Ariane captureraient les URL et requetes (jetons de reset, ids de job).
        maxBreadcrumbs: 0,
        beforeSend(event) {
          // /reset-password?token=... : ne jamais envoyer la query string
          if (event.request?.url) event.request.url = event.request.url.split("?")[0];
          return event;
        },
      });
    });
  }, []);
  return null;
}
