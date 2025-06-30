import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def enviar_email(destinatario, assunto, corpo_html):
    # CONFIGURAÇÕES SMTP HOSTINGER
    smtp_host = "smtp.hostinger.com"
    smtp_port = 465
    smtp_user = "contact@learningbyworking.com"
    smtp_pass = "Dani4035*+"

    # MONTAGEM DO EMAIL
    msg = MIMEMultipart("alternative")
    msg["Subject"] = assunto
    msg["From"] = smtp_user
    msg["To"] = destinatario

    parte_html = MIMEText(corpo_html, "html")
    msg.attach(parte_html)

    # ENVIO SMTP
    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, destinatario, msg.as_string())
        print("✅ Email enviado com sucesso.")
        return True
    except Exception as e:
        print(f"❌ Erro ao enviar email: {e}")
        return False

# EXEMPLO DE TESTE DIRETO
if __name__ == "__main__":
    destinatario = "seuemail@gmail.com"  # altere para testar
    assunto = "Redefinição de Senha - Learning by Working"
    corpo_html = """
    <html>
    <body>
      <h2>Olá!</h2>
      <p>Segue o link para redefinir sua senha:</p>
      <a href='https://seusite.com/reset'>Clique aqui para redefinir</a>
      <p>Se não solicitou, ignore este email.</p>
      <p>Atenciosamente,<br><b>Learning by Working</b></p>
    </body>
    </html>
    """
    enviar_email(destinatario, assunto, corpo_html)
