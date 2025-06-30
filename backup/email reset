import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def enviar_email_reset(destinatario_email, link_reset):
    remetente = "contact@learningbyworking.com"
    senha_remetente = "Dani4035*+"
    servidor_smtp = "smtp.hostinger.com"
    porta_smtp = 465

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Recuperação de Senha – Learning by Working"
    msg["From"] = remetente
    msg["To"] = destinatario_email

    html = f"""
    <html>
      <body>
        <p>Olá,</p>
        <p>Você solicitou a redefinição da sua senha. Clique no botão abaixo:</p>
        <p><a href="{link_reset}" style="padding:10px 20px; background:#4CAF50; color:white; text-decoration:none; border-radius:5px;">Redefinir Senha</a></p>
        <p>Se não solicitou, ignore este e-mail.</p>
        <br>
        <p>Equipe Learning by Working</p>
      </body>
    </html>
    """

    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL(servidor_smtp, porta_smtp) as server:
            server.login(remetente, senha_remetente)
            server.sendmail(remetente, destinatario_email, msg.as_string())
            print("✅ E-mail enviado com sucesso.")
    except Exception as e:
        print("❌ Erro ao enviar e-mail:", e)


        import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def enviar_email_reset(destinatario, link_reset):
    # CONFIGURAÇÕES SMTP HOSTINGER
    smtp_host = "smtp.hostinger.com"
    smtp_port = 465
    smtp_user = "contact@learningbyworking.com"
    smtp_pass = "Dani4035*+"

    # MONTAGEM DO EMAIL
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Redefinição de Senha - Learning by Working"
    msg["From"] = smtp_user
    msg["To"] = destinatario

    corpo_html = f"""
    <html>
    <body>
      <h2>Olá!</h2>
      <p>Segue o link para redefinir sua senha:</p>
      <a href='{link_reset}'>Clique aqui para redefinir</a>
      <p>Se não solicitou, ignore este email.</p>
      <p>Atenciosamente,<br><b>Learning by Working</b></p>
    </body>
    </html>
    """

    parte_html = MIMEText(corpo_html, "html")
    msg.attach(parte_html)

    # ENVIO SMTP
    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, destinatario, msg.as_string())
        print("✅ Email de reset enviado com sucesso.")
        return True
    except Exception as e:
        print(f"❌ Erro ao enviar email de reset: {e}")
        return False
