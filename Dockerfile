FROM python:3.9-slim

EXPOSE 8501

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./arquivos/ /usr/src/app
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
WORKDIR /usr/src/app

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=127.0.0.1"]
