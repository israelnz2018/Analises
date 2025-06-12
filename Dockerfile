FROM python:3.11-slim

# 1) Instala dependências Python
WORKDIR /app
COPY n8n/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# 2) Copia seu código (FastAPI + HTMLs + start.sh)
COPY n8n/ /app/n8n/
COPY n8n/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# 3) Instala Node.js e n8n (versão estável e compatível)
RUN apt-get update \
 && apt-get install -y curl gnupg \
 && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
 && apt-get install -y nodejs \
 && npm install -g n8n@1.36.2 \
 && rm -rf /var/lib/apt/lists/*

# 4) Define variável de build e ambiente
ARG PROJECT=fastapi
ENV PROJECT=${PROJECT}

# 5) Expõe portas usadas
EXPOSE 8000 5678

# 6) Ponto de entrada
ENTRYPOINT ["/app/start.sh"]




