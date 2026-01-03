# Pick image for running Python 
FROM python:3.12-slim

# Meta-data
LABEL maintainer="m.astashonak@gmail.com"

# Python globals
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies 
# - gcc & build-essentials - can be used for dependencies compilation
# - libpq-dev - headers for PostgreSQL (for psycopg2)
# - postgresql postgresql-contrib - PostgreSQL server
# - curl - for debugging (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev curl postgresql postgresql-contrib \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy application sources
COPY app/ /app/

# Installing dependencies 
RUN pip install --no-cache-dir fastapi uvicorn[standard] streamlit catboost \
    pandas numpy scikit-learn sqlalchemy psycopg2-binary python-dotenv pyyaml

# Ports: FastAPI (8000), Streamlit (8501), PostgreSQL (5432)
EXPOSE 8000 8501 5432

# Note: PostgreSQL cluster will be auto-initialized on first service start

# Set environment variables for database
ENV DATABASE_NAME=credit_scoring \
    DATABASE_USER=credit_user \
    DATABASE_PASSWORD=credit_pass \
    DATABASE_HOST=localhost \
    DATABASE_PORT=5432

# Run script
RUN printf '#!/usr/bin/env bash\n'\
    'set -e\n'\
    'echo "Starting PostgreSQL..."\n'\
    'service postgresql start || /usr/lib/postgresql/*/bin/pg_ctlcluster * main start || true\n'\
    'sleep 5\n'\
    'echo "Creating database and user if not exists..."\n'\
    'su postgres -c "psql -c \\"CREATE USER credit_user WITH SUPERUSER PASSWORD '\''credit_pass'\'';\\"" 2>/dev/null || echo "User already exists"\n'\
    'su postgres -c "createdb -O credit_user credit_scoring" 2>/dev/null || echo "Database already exists"\n'\
    'echo "Waiting for PostgreSQL to be ready..."\n'\
    'for i in {1..30}; do\n'\
    '  if pg_isready -U postgres >/dev/null 2>&1; then\n'\
    '    break\n'\
    '  fi\n'\
    '  sleep 1\n'\
    'done\n'\
    'echo "Loading database schema..."\n'\
    'if [ -f /app/credit_scoring.sql ]; then\n'\
    '  PGPASSWORD=credit_pass psql -U credit_user -d credit_scoring -f /app/credit_scoring.sql 2>/dev/null || echo "SQL file already loaded or error occurred"\n'\
    'fi\n'\
    'echo "Starting FastAPI (server.py) on :8000..."\n'\
    'cd /app && uvicorn server:app --host 0.0.0.0 --port 8000 &\n'\
    'echo "Starting Streamlit (client.py) on :8501..."\n'\
    'cd /app && streamlit run client.py --server.port 8501 --server.address 0.0.0.0\n' \
    > /app/start.sh && chmod +x /app/start.sh

# 9. Команда запуска контейнера 
CMD ["bash", "/app/start.sh"]