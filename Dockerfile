FROM public.ecr.aws/docker/library/python:3.11-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter

ENV AWS_LWA_PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_MODEL_PATH=/opt/models/bge-small-en-v1.5 \
    HF_HOME=/tmp/huggingface

WORKDIR /var/task

# CPU-only torch first: it is the biggest layer and changes least.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/download_embedding_model.py scripts/
RUN python scripts/download_embedding_model.py

# From here on the image must never reach for the internet.
ENV HF_HUB_OFFLINE=1

COPY app/ app/

CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]