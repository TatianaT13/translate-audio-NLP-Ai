/**
 * Bibliotheque d'icones SVG utilisees dans l'app. Style outline uniforme
 * (stroke-width 1.75, currentColor) pour heriter des couleurs CSS et rester
 * coherent avec le design minimaliste du reste de l'app.
 */
import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function base(props: IconProps) {
  const { size = 24, ...rest } = props;
  return {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.75,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    ...rest,
  };
}

// ── Feature icons ────────────────────────────────────────────────────────────

/** Traduction : deux fleches echangees + petit trait audio pour rappeler la voix. */
export function TranslateIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M4 7h9M4 7l3-3M4 7l3 3" />
      <path d="M20 17h-9m9 0l-3-3m3 3l-3 3" />
      <path d="M3 14v1M6 13v3M9 14v1" opacity="0.5" />
    </svg>
  );
}

/** Live : microphone stylise. */
export function MicIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <line x1="12" y1="18" x2="12" y2="22" />
      <line x1="9" y1="22" x2="15" y2="22" />
    </svg>
  );
}

/** Reunion : document avec lignes + petites puces pour represener des notes. */
export function DocumentIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M6 3h9l4 4v14a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z" />
      <path d="M15 3v4h4" />
      <circle cx="8.5" cy="12" r="0.5" fill="currentColor" />
      <line x1="10" y1="12" x2="16" y2="12" />
      <circle cx="8.5" cy="15" r="0.5" fill="currentColor" />
      <line x1="10" y1="15" x2="16" y2="15" />
      <circle cx="8.5" cy="18" r="0.5" fill="currentColor" />
      <line x1="10" y1="18" x2="14" y2="18" />
    </svg>
  );
}

// ── UI icons ─────────────────────────────────────────────────────────────────

export function HomeIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M3 11l9-8 9 8" />
      <path d="M5 10v10a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1V10" />
    </svg>
  );
}

export function AdminIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M12 2l9 4v6c0 5-4 9-9 10-5-1-9-5-9-10V6l9-4z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  );
}

export function LogoutIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
      <polyline points="16 17 21 12 16 7" />
      <line x1="21" y1="12" x2="9" y2="12" />
    </svg>
  );
}

export function KeyIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="7" cy="14" r="4" />
      <path d="M10 12l10-10" />
      <path d="M15 7l3 3" />
      <path d="M18 4l3 3" />
    </svg>
  );
}

export function TrashIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  );
}

export function ChevronDownIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

export function ExternalLinkIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  );
}

export function CardIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <rect x="2" y="6" width="20" height="14" rx="2" />
      <line x1="2" y1="11" x2="22" y2="11" />
      <line x1="6" y1="16" x2="10" y2="16" />
    </svg>
  );
}
