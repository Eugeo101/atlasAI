FROM python:3.12-slim

WORKDIR /app

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy main.py relative to ./services/gateway context
COPY main.py .

EXPOSE 8080

# MUST bind to 0.0.0.0 inside container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]