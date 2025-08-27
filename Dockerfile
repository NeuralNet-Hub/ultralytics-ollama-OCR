# Use the official ultralytics image as base
FROM ultralytics/ultralytics:latest

# Set working directory
WORKDIR /app

# Copy requirements and install additional packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all application files
COPY . .

# Expose Gradio port
EXPOSE 7860

# Set default command with your ALPR model
CMD ["python", "main.py", "--model", "alpr-yolo11s-aug.pt"]
