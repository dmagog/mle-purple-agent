FROM python:3.11-slim

WORKDIR /app

# System deps for ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install heavy ML libs first (separate layer for better caching)
RUN pip install --no-cache-dir --timeout 300 \
    numpy>=1.26.0 \
    pandas>=2.0.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.4.0

RUN pip install --no-cache-dir --timeout 300 \
    xgboost>=2.0.0 \
    lightgbm>=4.3.0

# Install framework + API deps
RUN pip install --no-cache-dir --timeout 300 \
    "a2a-sdk[http-server]>=0.3.0" \
    openai>=1.30.0 \
    uvicorn>=0.29.0 \
    python-dotenv>=1.0.0

COPY src/ ./src/

WORKDIR /app/src

EXPOSE 8000

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
