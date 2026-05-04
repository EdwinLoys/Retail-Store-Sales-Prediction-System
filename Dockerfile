# Use a minimal Python image
FROM python:3.11-slim

# Create a non‑root user for safety
RUN useradd -m appuser
WORKDIR /app
USER appuser

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ src/
COPY app.py .
COPY run.py .
COPY pyproject.toml .

# Install the package in editable mode so the console‑script is available
RUN pip install -e .

# Default entry point – runs the console script defined in pyproject.toml
ENTRYPOINT ["app"]