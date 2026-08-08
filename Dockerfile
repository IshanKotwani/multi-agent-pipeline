# Slim base — small image, matches your local Python 3.12
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# Copy requirements ALONE first, install, THEN copy code so Docker's
# layer cache doesn't reinstall deps every time you edit source.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data dir and run as non-root (security best practice)
RUN mkdir -p ${DATA_DIR} && useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

VOLUME ["/app/data"]
EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py"]
