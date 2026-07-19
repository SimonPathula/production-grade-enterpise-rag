FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements-backend.txt

COPY app/ ./app/

ENV PYTHONBUFFERED=1
ENV PORT=8000

EXPOSE 8000

CMD [ "python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]