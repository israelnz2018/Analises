import firebase_admin
from firebase_admin import auth, credentials
from email_reset import enviar_email_reset  # ajuste o nome do arquivo se necessário

# ✅ Inicializa o Firebase apenas uma vez no projeto
try:
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate("firebase_service_account.json")
    firebase_admin.initialize_app(cred)


def gerar_link_reset_e_enviar(destinatario):
    """
    Gera o link de redefinição de senha no Firebase
    e envia um email bonito para o destinatário.
    """
    try:
        # 🔗 Gera link de reset
        link_reset = auth.generate_password_reset_link(destinatario)
        print("🔗 Link de reset gerado:", link_reset)

        # ✉️ Envia email customizado
        enviado = enviar_email_reset(destinatario, link_reset)
        return enviado

    except Exception as e:
        print(f"❌ Erro ao gerar link ou enviar email: {e}")
        return False
