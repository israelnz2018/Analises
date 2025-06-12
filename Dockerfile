FROM python:3.11-slim

# 1) Instala dependências Python
WORKDIR /app
COPY n8n/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# 2) Copia seu código (FastAPI + start.sh)
COPY n8n/ /app/n8n/
COPY n8n/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# 3) (Opcional) Instala Node.js e n8n CLI – pode manter se você ainda usa algum fluxo no terminal
RUN apt-get update \
 && apt-get install -y curl gnupg \
 && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
 && apt-get install -y nodejs \
 && npm install -g n8n \
 && rm -rf /var/lib/apt/lists/*

# 4) Define o projeto como Analises
ARG PROJECT=analises
ENV PROJECT=${PROJECT}

# 5) Expõe as portas padrão
EXPOSE 8000 5678

# 6) Inicia o script de entrada
ENTRYPOINT ["/app/start.sh"]







