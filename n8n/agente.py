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
def perguntar_ia(pergunta: str, tipo: str) -> str:
    # Recupera contexto anterior
    contexto = recuperar_ultimas_conversas(tipo, limite=5)

    # Monta prompt completo
    prompt = (
        "Você é um assistente estatístico especializado. "
        "Aqui está o histórico recente:\n\n"
    )
    for idx, item in enumerate(contexto, 1):
        prompt += f"[{idx}] {item}\n\n"

    prompt += f"Pergunta do aluno: {pergunta}\n\n"
    prompt += "Responda de forma prática, clara e objetiva, sem linguagem técnica excessiva."

    # Chama OpenAI
    resposta = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Você é um especialista em estatística e análise de gráficos."},
            {"role": "user", "content": prompt}
        ]
    )

    resposta_final = resposta.choices[0].message.content

    # Salva pergunta e resposta no histórico
    salvar_conversa(tipo, f"Pergunta: {pergunta}\nResposta: {resposta_final}")

    return resposta_final


