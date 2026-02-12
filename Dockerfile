# Use slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for TensorFlow etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements/requirements.txt .

RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy important project folders
# COPY app/ ./app/
COPY app/ ./app/
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY config.yaml .
COPY main.py .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
