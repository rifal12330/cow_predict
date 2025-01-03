# Gunakan base image Python
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Copy semua file ke container
COPY . .

# Install dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Flask
EXPOSE 8080

# Jalankan aplikasi Flask
CMD ["python", "app.py"]
