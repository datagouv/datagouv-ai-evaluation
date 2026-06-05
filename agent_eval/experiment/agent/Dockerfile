FROM python:3.12-slim
ARG DATAGOUV_CLI=false
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl wget jq unzip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir requests httpx pandas
RUN if [ "$DATAGOUV_CLI" = "true" ]; then pip install --no-cache-dir datagouv-client; fi
