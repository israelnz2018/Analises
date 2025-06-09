#!/bin/sh
echo "🚀 Startup: PROJECT=$PROJECT"
echo "📁 Diretório atual: $(pwd)"

if [ "$PROJECT" = "analises" ]; then
  echo "▶️ Modo Análises: iniciando FastAPI..."
  cd /app/n8n
  uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

elif [ "$PROJECT" = "html" ]; then
  echo "▶️ Modo HTML/API: iniciando n8n Workflow Designer..."
  n8n start
else
  echo "❌ Valor inválido para PROJECT: $PROJECT"
  echo "   Use PROJECT=html ou PROJECT=analises"
  exit 1
fi














