# Use PyTorch base image with Python 3.9
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip/*

# Copy the application code and model
COPY . .

# Create model directory if it doesn't exist
RUN mkdir -p scripts/outputs

# Expose Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
