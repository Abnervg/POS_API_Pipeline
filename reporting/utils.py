import logging
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os

def convert_md_to_pdf(md_path, output_dir):
    """
    Converts a Markdown report file to a PDF.

    Args:
        md_path (Path): The path to the input .md file.
        output_dir (Path): The directory to save the output .pdf file.

    Returns:
        Path: The path to the generated PDF file, or None on failure.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Converting {md_path} to PDF...")
    
    try:
        pdf = MarkdownPdf()
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # The root_path tells the converter where to find the embedded images
        pdf.add_section(Section(md_content, root_path=md_path.parent))
        
        pdf_path = output_dir / f"{md_path.stem}.pdf"
        pdf.save(pdf_path)
        
        logger.info(f"Successfully converted report to PDF: {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.error(f"Failed to convert Markdown to PDF. Error: {e}")
        return None

def send_report_by_email(pdf_path, recipient_email, file_tag):
    """
    Sends the generated PDF report as an email attachment.
    """
    logger = logging.getLogger(__name__)
    
    # Get SMTP configuration from environment variables (set by GitHub Secrets)
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([smtp_host, smtp_port, smtp_user, smtp_password]):
        logger.error("SMTP configuration is missing. Cannot send email.")
        return

    logger.info(f"Preparing to send report to {recipient_email}...")

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = recipient_email
    msg['Subject'] = f"Monthly Sales Report: {file_tag}"

    # Add the email body
    body = f"Please find the attached sales report for {file_tag}."
    msg.attach(MIMEText(body, 'plain'))

    # Attach the PDF file
    with open(pdf_path, "rb") as f:
        attach = MIMEApplication(f.read(), _subtype="pdf")
    attach.add_header('Content-Disposition', 'attachment', filename=str(pdf_path.name))
    msg.attach(attach)

    # Send the email
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        logger.info("Email sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send email. Error: {e}")

