FROM python:3.11-slim

WORKDIR /app

# Copy dependencies ก่อน
COPY requirements.txt .

# ติดตั้ง Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config.yaml .

CMD ["python", "src/train_V7.py"]