# Portable image — works on Fly.io, Railway, Google Cloud Run, or any container host.
FROM python:3.11-slim

WORKDIR /app

# System libs some scientific wheels expect at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The host injects $PORT; default to 8090 for plain `docker run`.
ENV PORT=8090
EXPOSE 8090

# Seed logs/ (idempotent) then start the single-worker production server.
CMD ["sh", "-c", "python bootstrap.py && gunicorn -w 1 -b 0.0.0.0:${PORT} app_user:app"]
