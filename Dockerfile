# Dockerfile exclusivo para N8N
FROM node:18

# Cria diretório de trabalho
WORKDIR /data

# Instala n8n
RUN npm install -g n8n

# Expõe a porta padrão
EXPOSE 5678

# Inicia o n8n
CMD ["n8n"]





