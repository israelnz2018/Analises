import requests

def perguntar_ia(pergunta: str, analise: str) -> str:
    url = "https://primary-production-1d53.up.railway.app/webhook/perguntar-ia"

    payload = {
        "prompt": pergunta,
        "analise": analise
    }

    try:
        resposta = requests.post(url, json=payload)
        resposta.raise_for_status()
        dados = resposta.json()

        return dados.get("analise", "⚠️ Nenhuma resposta recebida do agente.")
    except Exception as e:
        return f"❌ Erro ao consultar o agente: {str(e)}"

