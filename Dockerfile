# --- Stage 1: Builder ---
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: Final Runtime ---
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed pip packages from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Set up environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Expose port for local dashboard server
EXPOSE 8000

# Directory for output reports volume
VOLUME [ "/app/reports" ]

# Default CLI entrypoint
ENTRYPOINT [ "python", "src/main.py" ]
CMD [ "run", "--suite", "golden", "--model", "mock" ]
