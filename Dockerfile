# Use Python 3.13 slim image as base
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install UV package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using UV
RUN uv sync --frozen --no-cache

# Copy application code
COPY . .

# Create a directory for JSON files if needed
RUN mkdir -p /app/data

# Expose the port Flask runs on
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Default command - run Flask app with default JSON file
# Users can override the JSON file by mounting a volume and changing the command
CMD ["uv", "run", "python", "main.py", "--file", "test.json"]
