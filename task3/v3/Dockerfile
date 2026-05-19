# Use official lightweight Python image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install system dependencies needed for compiling certain python libs (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to utilize Docker build layer caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and orchestrator
COPY src/ ./src/
COPY main.py .

# Run the predictive maintenance ML pipeline
CMD ["python", "main.py"]
