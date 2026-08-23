from __future__ import annotations

from resume_checker.config import get_settings


def send_html_email(html_content: str, receiver_email: str) -> str:
    settings = get_settings()
    if not settings.sendgrid_api_key:
        return "skipped: SENDGRID_API_KEY is not configured"
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
    except ImportError:
        return "skipped: sendgrid extra is not installed"

    message = Mail(
        from_email=settings.from_email,
        to_emails=receiver_email,
        subject="Resume analysis report",
        html_content=html_content,
    )
    client = SendGridAPIClient(settings.sendgrid_api_key)
    response = client.send(message)
    return f"sent:{response.status_code}"
