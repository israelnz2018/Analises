#!/bin/sh
echo "🟡 [start.sh] PROJECT=${PROJECT}"
echo "📁 Diretório atual: $(pwd)"

if [ "$PROJECT" = "html" ]; then
  echo "📂 Conteúdo de /app/n8n:"
  ls -la /app/n8n

  echo "📄 Verificando se /app/n8n/main.py existe..."
  if [ -f /app/n8n/main.py ]; then
    echo "✅ /app/n8n/main.py encontrado!"
  else
    echo "❌ ERRO: /app/n8n/main.py NÃO encontrado!"
    exit 1
  fi

  echo "🚀 Iniciando FastAPI (formulário) em $PORT ou fallback 8000..."
  cd /app/n8n
  uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

else
  echo "🚀 Iniciando n8n Workflow Designer em ${PORT}..."
  n8n start --host 0.0.0.0 --port ${PORT}
fi















