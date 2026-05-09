FROM python:3.11-slim 

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/n8n

COPY n8n/requirements.txt requirements.txt
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY n8n/ /app/n8n/
RUN chmod +x start.sh

ARG PROJECT=html
ENV PROJECT=${PROJECT}

EXPOSE 8000

ENTRYPOINT ["./start.sh"]







