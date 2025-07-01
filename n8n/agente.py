# agente.py - atualizado para uso direto com OpenAI + memória SQLite

import sqlite3
from openai import OpenAI

# Inicializa cliente OpenAI
client = OpenAI()

# ✅ Função para inicializar banco SQLite
def init_db():
    conn = sqlite3.connect("memoria_ia.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo TEXT,        -- 'analise' ou 'grafico'
            conteudo TEXT     -- texto ou base64 do grafico
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ✅ Função para salvar conversa
def salvar_conversa(tipo: str, conteudo: str):
    conn = sqlite3.connect("memoria_ia.db")
    c = conn.cursor()
    c.execute('INSERT INTO conversas (tipo, conteudo) VALUES (?, ?)', (tipo, conteudo))
    conn.commit()
    conn.close()

# ✅ Função para recuperar as últimas N conversas
def recuperar_ultimas_conversas(tipo: str, limite: int = 5):
    conn = sqlite3.connect("memoria_ia.db")
    c = conn.cursor()
    c.execute('SELECT conteudo FROM conversas WHERE tipo=? ORDER BY id DESC LIMIT ?', (tipo, limite))
    resultados = c.fetchall()
    conn.close()
    return [r[0] for r in resultados][::-1]  # retorna na ordem cronológica

# ✅ Função principal de pergunta IA
def perguntar_ia(pergunta: str, texto_analise: str) -> str:
    # 🔧 Usa diretamente a última análise recebida como contexto principal
    prompt = (
        "Você é um analista estatístico especialista em análises de dados reais.\n\n"
        "Aqui estão os resultados da última análise realizada pelo sistema:\n\n"
        f"{texto_analise}\n\n"
        "Baseado nestes resultados, responda de forma direta e técnica à pergunta do aluno, "
        "interpretando os dados apresentados sem explicações genéricas de estatística. "
        "Fale como um analista responderia a um engenheiro de processos.\n\n"
        "Se necessrio, recomende o uso de alguma ferramenta grafica ou estatistica adicional par auxiliar o aluno.\n\n"
        f"Pergunta do aluno: {pergunta}"
    )

    # Chama OpenAI
    resposta = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Você é um analista estatístico especialista em análises de processos."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    resposta_final = resposta.choices[0].message.content

    # 🔧 Opcional: salvar pergunta e resposta no histórico (ajuste se quiser manter histórico separado)
    salvar_conversa("analise", f"Pergunta: {pergunta}\nResposta: {resposta_final}")

    return resposta_final




