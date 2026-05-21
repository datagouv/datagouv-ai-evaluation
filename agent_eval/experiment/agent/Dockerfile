FROM python:3.12-slim
ARG DATAGOUV_CLI=false
RUN pip install --no-cache-dir requests httpx pandas
RUN if [ "$DATAGOUV_CLI" = "true" ]; then pip install --no-cache-dir datagouv-client; fi
