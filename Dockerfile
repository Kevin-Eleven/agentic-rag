FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY agents ./agents
COPY config ./config
COPY llm ./llm
COPY retrieval ./retrieval
COPY utils ./utils
COPY workflow ./workflow
COPY ui ./ui
COPY eval ./eval
COPY main.py ./main.py
COPY .streamlit ./.streamlit

RUN pip install --no-cache-dir -e ".[ui]"

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0"]
