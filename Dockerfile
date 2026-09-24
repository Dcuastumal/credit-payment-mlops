FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["python", "-m", "uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
