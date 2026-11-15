# MediStock AI — API image
#
# Build:  docker build -t medistock-api .
# Run:    docker run -p 8000:8000 -v ${PWD}/artifacts:/app/artifacts:ro medistock-api
#
# The model artifact is mounted read-only rather than baked into the image, so a
# retrain does not require rebuilding the container and so the served model is
# auditable (the manifest is visible in the mounted directory).

FROM python:3.11-slim

LABEL org.opencontainers.image.title="MediStock AI" \
      org.opencontainers.image.description="Pharmacy demand forecasting and expiry-risk API" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MEDISTOCK_ENVIRONMENT=production \
    MEDISTOCK_LOG_JSON=true

WORKDIR /app

# Dependencies first: this layer is cached until the lock file changes.
COPY requirements.lock.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.lock.txt

COPY pyproject.toml README.md ./
COPY app ./app
COPY ml ./ml
COPY scripts ./scripts
COPY dashboard ./dashboard

# Non-root: the service only needs to read artifacts and write logs.
RUN useradd --create-home --uid 10001 medistock \
    && mkdir -p /app/artifacts /app/data /app/reports/logs \
    && chown -R medistock:medistock /app
USER medistock

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
