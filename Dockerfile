FROM python:3.11-slim

# Install system deps for pymupdf / pillow
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libjpeg-dev zlib1g-dev libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip setuptools wheel
RUN pip install .

# Default command: run pipeline entrypoint (override with docker run args)
ENTRYPOINT ["python", "pipeline.py"]

CMD ["--pdf", "/app/pde2025.pdf", "--start-page", "1", "--output", "/app/output.tex"]