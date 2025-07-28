FROM --platform=linux/amd64 python:3.13

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK resources
RUN python -m nltk.downloader punkt averaged_perceptron_tagger

# Copy application files
COPY process_pdfs.py .
COPY model_RandomForest.pkl .

# Set default command
CMD ["python", "process_pdfs.py"]
