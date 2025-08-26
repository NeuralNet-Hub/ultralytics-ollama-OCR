# Use the official ultralytics image as base
FROM ultralytics/ultralytics:latest

# Set working directory
WORKDIR /app

# Copy requirements and install additional packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the main application
COPY main.py .

# Expose Gradio port
EXPOSE 7860

# Set default command
CMD ["python", "main.py", "--model", "yolo11n.pt"]
