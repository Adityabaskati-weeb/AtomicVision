FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir .

EXPOSE 7860

CMD ["server"]
