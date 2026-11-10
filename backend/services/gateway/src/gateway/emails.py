"""Emails transactionnels via Resend (API HTTP, pas de SDK).

Sans RESEND_API_KEY, les envois sont des no-op logués : un email raté ne
doit jamais bloquer une inscription ou un paiement.
"""
from __future__ import annotations

import os

import httpx

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
EMAIL_FROM     = os.getenv("EMAIL_FROM", "traduction-audio.fr <noreply@traduction-audio.fr>")
APP_BASE_URL   = os.getenv("APP_BASE_URL", "https://traduction-audio.fr")
SUPPORT_EMAIL  = os.getenv("SUPPORT_EMAIL", "contact@traduction-audio.fr")

RESEND_ENDPOINT = "https://api.resend.com/emails"


def _send(to: str, subject: str, html: str, text: str) -> None:
    if not RESEND_API_KEY:
        print(f"[emails] RESEND_API_KEY vide, no-op subject={subject!r}", flush=True)
        return
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.post(
                RESEND_ENDPOINT,
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={"from": EMAIL_FROM, "to": [to], "subject": subject,
                      "html": html, "text": text, "reply_to": SUPPORT_EMAIL},
            )
        if r.status_code >= 400:
            print(f"[emails] Resend {r.status_code}: {r.text[:200]}", flush=True)
        else:
            print(f"[emails] sent subject={subject!r}", flush=True)
    except Exception as e:
        print(f"[emails] send error: {e}", flush=True)


_WRAPPER_OPEN = """<!doctype html><html lang="fr"><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#0f0e0c;font-family:Georgia,serif;color:#f5efe4;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#0f0e0c;padding:40px 20px;">
<tr><td align="center">
<table width="560" cellpadding="0" cellspacing="0" style="background:#1a1612;border:1px solid #2a241e;border-radius:12px;">
<tr><td style="padding:36px 40px;">
<div style="font-size:13px;color:#c9a96e;letter-spacing:2px;margin-bottom:28px;">TRADUCTION-AUDIO.FR</div>
"""
_WRAPPER_CLOSE = """
</td></tr>
<tr><td style="padding:20px 40px 28px;border-top:1px solid #2a241e;font-size:12px;color:#8a8578;line-height:1.6;">
traduction-audio.fr · <a href="mailto:{support}" style="color:#c9a96e;text-decoration:none;">{support}</a>
<br>Vous recevez cet email car vous avez un compte sur traduction-audio.fr.
</td></tr>
</table>
</td></tr>
</table>
</body></html>"""

_P   = 'style="font-size:15px;line-height:1.7;color:#d8d1c2;margin:0 0 16px;"'
_H1  = 'style="font-size:24px;font-weight:400;margin:0 0 20px;line-height:1.3;"'
_SM  = 'style="font-size:13px;color:#8a8578;line-height:1.6;margin:24px 0 0;"'
_BTN = 'style="display:inline-block;background:#c9a96e;color:#0f0e0c;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:500;"'


def _wrap(body_html: str) -> str:
    return _WRAPPER_OPEN + body_html + _WRAPPER_CLOSE.format(support=SUPPORT_EMAIL)


def _button(href: str, label: str) -> str:
    return f'<p style="margin:28px 0;"><a href="{href}" {_BTN}>{label}</a></p>'


def send_welcome(to: str) -> None:
    subject = "Bienvenue sur traduction-audio.fr"
    body = (
        f"<h1 {_H1}>Bienvenue.</h1>"
        f"<p {_P}>Votre compte est actif. <strong style=\"color:#c9a96e;\">10 crédits offerts</strong> "
        "vous attendent pour découvrir le service :</p>"
        '<ul style="font-size:14px;line-height:1.9;color:#d8d1c2;margin:0 0 24px;padding-left:20px;">'
        "<li>Transcription et traduction de fichiers audio, avec voix de synthèse</li>"
        "<li>Comptes-rendus de réunion avec identification des intervenants</li>"
        "<li>Traduction vocale en temps réel</li>"
        "</ul>"
        + _button(f"{APP_BASE_URL}/translate", "Commencer")
        + f"<p {_SM}>Une question ? Répondez simplement à cet email.</p>"
    )
    text = (
        "Bienvenue sur traduction-audio.fr.\n\n"
        "Votre compte est actif. 10 crédits offerts vous attendent pour découvrir :\n"
        "- la transcription et traduction de fichiers audio, avec voix de synthèse\n"
        "- les comptes-rendus de réunion avec identification des intervenants\n"
        "- la traduction vocale en temps réel\n\n"
        f"Commencer : {APP_BASE_URL}/translate\n\n"
        "Une question ? Répondez simplement à cet email.\n"
    )
    _send(to, subject, _wrap(body), text)


def send_verification(to: str, verify_url: str) -> None:
    subject = "Confirmez votre adresse email"
    body = (
        f"<h1 {_H1}>Confirmez votre adresse</h1>"
        f"<p {_P}>Un clic pour activer votre compte et recevoir vos "
        '<strong style="color:#c9a96e;">10 crédits offerts</strong>. Ce lien est valable '
        "<strong>48 heures</strong>.</p>"
        + _button(verify_url, "Confirmer mon adresse")
        + f"<p {_SM}>Si vous n'avez pas créé de compte sur traduction-audio.fr, ignorez cet email.</p>"
        f'<p style="font-size:12px;color:#6a6558;margin:16px 0 0;word-break:break-all;">{verify_url}</p>'
    )
    text = (
        "Confirmez votre adresse email\n\n"
        "Un clic pour activer votre compte et recevoir vos 10 crédits offerts "
        "(lien valable 48 heures) :\n\n"
        f"{verify_url}\n\n"
        "Si vous n'avez pas créé de compte sur traduction-audio.fr, ignorez cet email.\n"
    )
    _send(to, subject, _wrap(body), text)


def send_reset_password(to: str, reset_url: str) -> None:
    subject = "Réinitialisation de votre mot de passe"
    body = (
        f"<h1 {_H1}>Réinitialisation de votre mot de passe</h1>"
        f"<p {_P}>Vous avez demandé à réinitialiser votre mot de passe. Ce lien est valable "
        "<strong>1 heure</strong>.</p>"
        + _button(reset_url, "Choisir un nouveau mot de passe")
        + f"<p {_SM}>Si vous n'êtes pas à l'origine de cette demande, ignorez cet email : "
        "votre mot de passe actuel reste inchangé.</p>"
        f'<p style="font-size:12px;color:#6a6558;margin:16px 0 0;word-break:break-all;">{reset_url}</p>'
    )
    text = (
        "Réinitialisation de votre mot de passe\n\n"
        "Vous avez demandé à réinitialiser votre mot de passe. Ce lien est valable 1 heure :\n\n"
        f"{reset_url}\n\n"
        "Si vous n'êtes pas à l'origine de cette demande, ignorez cet email.\n"
    )
    _send(to, subject, _wrap(body), text)


def send_payment_confirmation(to: str, monthly_credits: int) -> None:
    subject = "Paiement confirmé — votre abonnement Pro est actif"
    body = (
        f"<h1 {_H1}>Paiement confirmé.</h1>"
        f"<p {_P}>Merci ! Votre abonnement <strong style=\"color:#c9a96e;\">Pro</strong> est actif : "
        f"{monthly_credits} crédits sont alloués chaque mois, avec report des crédits non utilisés "
        "dans la limite du plafond.</p>"
        + _button(f"{APP_BASE_URL}/translate", "Utiliser mes crédits")
        + f"<p {_SM}>Vous pouvez résilier à tout moment depuis "
        f'<a href="{APP_BASE_URL}/consommation" style="color:#c9a96e;">Ma consommation</a>. '
        "Votre facture vous est envoyée séparément par Stripe.</p>"
    )
    text = (
        "Paiement confirmé.\n\n"
        f"Votre abonnement Pro est actif : {monthly_credits} crédits par mois, avec report "
        "des crédits non utilisés dans la limite du plafond.\n\n"
        f"Utiliser mes crédits : {APP_BASE_URL}/translate\n"
        f"Résilier à tout moment : {APP_BASE_URL}/consommation\n"
    )
    _send(to, subject, _wrap(body), text)


def send_renewal_notice(to: str, renewal_date: str) -> None:
    """Avis avant reconduction d'un abonnement annuel (art. L. 215-1 C. conso.) :
    la date limite de non-reconduction doit figurer dans un encadre apparent."""
    subject = f"Votre abonnement annuel sera renouvelé le {renewal_date}"
    box = (
        '<div style="border:2px solid #c9a96e;border-radius:10px;padding:16px 20px;margin:20px 0;'
        'font-size:15px;line-height:1.6;color:#f5efe4;">'
        f"<strong>Date limite pour ne pas renouveler : {renewal_date}</strong><br>"
        "Passé cette date, votre abonnement est reconduit pour un an au tarif en vigueur."
        "</div>"
    )
    body = (
        f"<h1 {_H1}>Votre abonnement arrive à échéance.</h1>"
        f"<p {_P}>Votre abonnement Pro annuel sera reconduit automatiquement le "
        f"<strong>{renewal_date}</strong> pour une nouvelle période d'un an.</p>"
        + box
        + f"<p {_P}>Si vous ne souhaitez pas le renouveler, vous pouvez le résilier gratuitement et à tout "
        "moment avant cette date. Il restera actif jusqu'à son terme.</p>"
        + _button(f"{APP_BASE_URL}/consommation", "Gérer ou résilier mon abonnement")
        + f"<p {_SM}>Vous n'avez rien à faire si vous souhaitez continuer.</p>"
    )
    text = (
        f"Votre abonnement Pro annuel sera reconduit automatiquement le {renewal_date} pour un an.\n\n"
        f"*** DATE LIMITE POUR NE PAS RENOUVELER : {renewal_date} ***\n\n"
        "Si vous ne souhaitez pas le renouveler, vous pouvez le résilier gratuitement avant cette date :\n"
        f"{APP_BASE_URL}/consommation\n\n"
        "Vous n'avez rien à faire si vous souhaitez continuer.\n"
    )
    _send(to, subject, _wrap(body), text)


def send_cancellation_confirmation(to: str, end_date: str | None) -> None:
    """Accuse de reception d'une resiliation, sur support durable (art. L. 215-1-1 C. conso.)."""
    until = f"le <strong>{end_date}</strong>" if end_date else "la fin de la période en cours"
    until_txt = f"le {end_date}" if end_date else "la fin de la période en cours"
    subject = "Résiliation de votre abonnement Pro enregistrée"
    body = (
        f"<h1 {_H1}>Résiliation enregistrée.</h1>"
        f"<p {_P}>Nous avons bien reçu votre demande de résiliation. Votre abonnement Pro reste actif "
        f"jusqu'à {until}, puis s'arrête : aucun nouveau prélèvement ne sera effectué.</p>"
        f"<p {_P}>Les crédits qu'il vous restera à cette date restent utilisables.</p>"
        + _button(f"{APP_BASE_URL}/consommation", "Voir mon abonnement")
        + f"<p {_SM}>Vous avez changé d'avis ? Vous pouvez reprendre votre abonnement depuis la même page "
        "tant qu'il n'est pas arrivé à son terme.</p>"
    )
    text = (
        "Résiliation enregistrée.\n\n"
        f"Votre abonnement Pro reste actif jusqu'à {until_txt}, puis s'arrête : aucun nouveau "
        "prélèvement ne sera effectué. Les crédits restants à cette date restent utilisables.\n\n"
        f"Voir mon abonnement : {APP_BASE_URL}/consommation\n"
    )
    _send(to, subject, _wrap(body), text)


def send_low_credits_alert(to: str, balance: int) -> None:
    subject = f"Il vous reste {balance} crédits"
    body = (
        f"<h1 {_H1}>Il vous reste {balance} crédits.</h1>"
        f"<p {_P}>Votre solde sera bientôt épuisé. Pour continuer sans interruption, "
        "découvrez l'abonnement Pro.</p>"
        + _button(f"{APP_BASE_URL}/tarifs", "Voir les formules")
    )
    text = (
        f"Il vous reste {balance} crédits.\n\n"
        "Votre solde sera bientôt épuisé. Pour continuer sans interruption :\n"
        f"{APP_BASE_URL}/tarifs\n"
    )
    _send(to, subject, _wrap(body), text)
