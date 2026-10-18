"""Emails transactionnels via Resend (resend.com).

API HTTP simple, pas de SDK → on utilise httpx deja present.
Si RESEND_API_KEY est vide, les envois sont no-op (dev / staging sans cle).

Les templates sont inline (HTML + plain-text) pour eviter un systeme de
templating en plus. Le ton et la mise en forme suivent la charte "elegante
sombre" du site : accent #c9a96e, texte dense, pas d'emoji dans le sujet.
"""
from __future__ import annotations

import os

import httpx

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
EMAIL_FROM     = os.getenv("EMAIL_FROM", "traduction-audio.fr <noreply@traduction-audio.fr>")
APP_BASE_URL   = os.getenv("APP_BASE_URL", "https://traduction-audio.fr")
SUPPORT_EMAIL  = os.getenv("SUPPORT_EMAIL", "contact@traduction-audio.fr")

RESEND_ENDPOINT = "https://api.resend.com/emails"


def _enabled() -> bool:
    return bool(RESEND_API_KEY)


def _send(to: str, subject: str, html: str, text: str) -> None:
    """Envoi fire-and-forget — toute erreur est loguee mais n'interrompt pas le flow.

    Un email rate ne doit JAMAIS casser une inscription ou un paiement.
    """
    if not _enabled():
        print(f"[emails] RESEND_API_KEY vide, no-op send to={to} subject={subject!r}", flush=True)
        return
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.post(
                RESEND_ENDPOINT,
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={"from": EMAIL_FROM, "to": [to], "subject": subject,
                      "html": html, "text": text},
            )
            if r.status_code >= 400:
                print(f"[emails] Resend {r.status_code} to={to}: {r.text[:200]}", flush=True)
            else:
                print(f"[emails] sent to={to} subject={subject!r}", flush=True)
    except Exception as e:
        print(f"[emails] send error to={to}: {e}", flush=True)


# ── Styles communs ────────────────────────────────────────────────────────────
_WRAPPER_OPEN = """<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#0f0e0c;font-family:ui-serif,Georgia,serif;color:#f5efe4;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#0f0e0c;padding:40px 20px;">
<tr><td align="center">
<table width="560" cellpadding="0" cellspacing="0" style="background:#1a1612;border:1px solid #2a241e;border-radius:12px;overflow:hidden;">
<tr><td style="padding:36px 40px;">
<div style="font-size:14px;color:#c9a96e;letter-spacing:2px;margin-bottom:28px;">TRADUCTION-AUDIO.FR</div>
"""
_WRAPPER_CLOSE = """
</td></tr>
<tr><td style="padding:20px 40px 28px;border-top:1px solid #2a241e;font-size:12px;color:#8a8578;line-height:1.6;">
traduction-audio.fr · Tetyana Tarasenko · <a href="mailto:{support}" style="color:#c9a96e;text-decoration:none;">{support}</a>
<br>Vous recevez cet email car vous avez un compte sur traduction-audio.fr.
</td></tr>
</table>
</td></tr>
</table>
</body></html>"""


def _wrap(body_html: str) -> str:
    return _WRAPPER_OPEN + body_html + _WRAPPER_CLOSE.format(support=SUPPORT_EMAIL)


# ── Templates ─────────────────────────────────────────────────────────────────

def send_welcome(to: str) -> None:
    """Email de bienvenue + 10 credits trial offerts."""
    subject = "Bienvenue sur traduction-audio.fr"
    body = f"""
<h1 style="font-size:26px;font-weight:400;margin:0 0 20px;line-height:1.3;">Bienvenue.</h1>
<p style="font-size:15px;line-height:1.7;color:#d8d1c2;margin:0 0 16px;">Votre compte est actif. Vous disposez de <strong style="color:#c9a96e;">10 credits offerts</strong> pour decouvrir toutes les fonctionnalites :</p>
<ul style="font-size:14px;line-height:1.9;color:#d8d1c2;margin:0 0 24px;padding-left:20px;">
<li>Transcription + traduction audio en 30 langues</li>
<li>Comptes-rendus de reunion avec identification des intervenants</li>
<li>Traduction vocale en temps reel</li>
</ul>
<p style="margin:28px 0;">
<a href="{APP_BASE_URL}/translate" style="display:inline-block;background:#c9a96e;color:#0f0e0c;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:500;">Commencer</a>
</p>
<p style="font-size:13px;color:#8a8578;line-height:1.6;margin:24px 0 0;">Une question ? Repondez simplement a cet email.</p>
"""
    text = (
        "Bienvenue sur traduction-audio.fr.\n\n"
        "Votre compte est actif. Vous avez 10 credits offerts pour decouvrir :\n"
        "- Transcription + traduction audio (30 langues)\n"
        "- Comptes-rendus de reunion avec diarisation\n"
        "- Traduction vocale en temps reel\n\n"
        f"Commencer : {APP_BASE_URL}/translate\n\n"
        f"Une question ? Repondez a cet email ou ecrivez a {SUPPORT_EMAIL}.\n"
    )
    _send(to, subject, _wrap(body), text)


def send_reset_password(to: str, reset_url: str) -> None:
    """Lien de reinitialisation (valide 1h)."""
    subject = "Reinitialisation de votre mot de passe"
    body = f"""
<h1 style="font-size:24px;font-weight:400;margin:0 0 20px;line-height:1.3;">Reinitialisation de votre mot de passe</h1>
<p style="font-size:15px;line-height:1.7;color:#d8d1c2;margin:0 0 24px;">Vous avez demande a reinitialiser votre mot de passe. Cliquez sur le bouton ci-dessous pour choisir un nouveau mot de passe. Ce lien expire dans <strong>1 heure</strong>.</p>
<p style="margin:24px 0;">
<a href="{reset_url}" style="display:inline-block;background:#c9a96e;color:#0f0e0c;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:500;">Reinitialiser mon mot de passe</a>
</p>
<p style="font-size:13px;color:#8a8578;line-height:1.6;margin:24px 0 0;">Si vous n'etes pas a l'origine de cette demande, ignorez simplement cet email : votre mot de passe actuel reste inchange.</p>
<p style="font-size:12px;color:#6a6558;margin:16px 0 0;word-break:break-all;">Lien direct : {reset_url}</p>
"""
    text = (
        "Reinitialisation de votre mot de passe\n\n"
        "Vous avez demande a reinitialiser votre mot de passe. Le lien ci-dessous expire dans 1 heure :\n\n"
        f"{reset_url}\n\n"
        "Si vous n'etes pas a l'origine de cette demande, ignorez cet email : votre mot de passe reste inchange.\n"
    )
    _send(to, subject, _wrap(body), text)


def send_payment_confirmation(to: str, tier: str, period_end_iso: str | None = None) -> None:
    """Confirmation apres checkout.session.completed."""
    subject = "Paiement confirme — bienvenue dans l'offre Pro"
    period_line = ""
    if period_end_iso:
        period_line = f"<p style=\"font-size:13px;color:#8a8578;margin:12px 0 0;\">Prochain renouvellement : {period_end_iso[:10]}.</p>"
    body = f"""
<h1 style="font-size:24px;font-weight:400;margin:0 0 20px;line-height:1.3;">Paiement confirme.</h1>
<p style="font-size:15px;line-height:1.7;color:#d8d1c2;margin:0 0 16px;">Merci — votre abonnement <strong style="color:#c9a96e;">{tier.upper()}</strong> est actif.</p>
<p style="font-size:14px;line-height:1.7;color:#d8d1c2;margin:0 0 20px;">Vos nouveaux credits sont immediatement disponibles : 200 credits processing + 10 credits live chaque mois, avec report plafonne.</p>
<p style="margin:28px 0;">
<a href="{APP_BASE_URL}/translate" style="display:inline-block;background:#c9a96e;color:#0f0e0c;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:500;">Utiliser mes credits</a>
</p>
<p style="font-size:13px;color:#8a8578;line-height:1.6;margin:24px 0 0;">Vous pouvez gerer votre abonnement (changement de carte, resiliation) depuis <a href="{APP_BASE_URL}/tarifs" style="color:#c9a96e;">votre espace tarifs</a>.</p>
{period_line}
"""
    text = (
        "Paiement confirme.\n\n"
        f"Votre abonnement {tier.upper()} est actif. Vous disposez de 200 credits processing + 10 credits live par mois.\n\n"
        f"Utiliser mes credits : {APP_BASE_URL}/translate\n"
        f"Gerer mon abonnement : {APP_BASE_URL}/tarifs\n"
    )
    if period_end_iso:
        text += f"\nProchain renouvellement : {period_end_iso[:10]}.\n"
    _send(to, subject, _wrap(body), text)


def send_low_credits_alert(to: str, balance: int, kind: str = "processing") -> None:
    """Alerte quand le solde passe sous le seuil (une seule fois par periode)."""
    subject = f"Solde faible — il vous reste {balance} credits"
    body = f"""
<h1 style="font-size:24px;font-weight:400;margin:0 0 20px;line-height:1.3;">Il vous reste {balance} credits.</h1>
<p style="font-size:15px;line-height:1.7;color:#d8d1c2;margin:0 0 16px;">Votre solde {kind} est bientot epuise. Pour continuer a utiliser traduction-audio.fr sans interruption, pensez a renouveler ou passer a l'offre Pro.</p>
<p style="margin:28px 0;">
<a href="{APP_BASE_URL}/tarifs" style="display:inline-block;background:#c9a96e;color:#0f0e0c;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:500;">Voir les formules</a>
</p>
<p style="font-size:13px;color:#8a8578;line-height:1.6;margin:24px 0 0;">Les credits non consommes sont reportes dans la limite du plafond mensuel.</p>
"""
    text = (
        f"Il vous reste {balance} credits {kind}.\n\n"
        "Pour continuer sans interruption, renouvelez ou passez a l'offre Pro :\n"
        f"{APP_BASE_URL}/tarifs\n"
    )
    _send(to, subject, _wrap(body), text)
