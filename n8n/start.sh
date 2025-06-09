#!/bin/sh
echo "🚀 Startup: PROJECT=$PROJECT"
echo "📁 Diretório atual: $(pwd)"

if [ "$PROJECT" = "html" ]; then
  # Projeto de formulário/API: levanta o FastAPI que serve HTML e /analise
  echo "▶️ Modo HTML/API: iniciando FastAPI..."
  cd /app/n8n
  uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

elif [ "$PROJECT" = "analises" ]; then
  # Projeto de workﬂow: levanta o n8n CLI para mostrar a UI de workflows
  echo "▶️ Modo Análises: iniciando n8n Workflow Designer..."
  n8n start --host 0.0.0.0 --port ${PORT:-5678}

else
  echo "❌ Valor inválido para PROJECT: $PROJECT"
  echo "   Use PROJECT=html ou PROJECT=analises"
  exit 1
fi











