FROM python:3.11-slim

WORKDIR /app

COPY n8n/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

COPY n8n/ /app/n8n/
RUN chmod +x /app/n8n/start.sh

ARG PROJECT=html
ENV PROJECT=${PROJECT}

EXPOSE 8000

ENTRYPOINT ["/app/n8n/start.sh"]








