FROM python:3.10-slim

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Download NLTK resources once during build
RUN python -m nltk.downloader punkt stopwords

# Expose port (Railway will bind automatically)
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]
