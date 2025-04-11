FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/usr/local/nltk_data

# Install basic system tools
RUN apt-get update && apt-get install -y build-essential

# Create working directory
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Pre-download NLTK data
RUN python -m nltk.downloader -d /usr/local/nltk_data stopwords

# Port for Railway/Heroku/etc
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
