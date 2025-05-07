# Step 1: Use an official PyTorch image as a parent image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    wget \
    unzip 
    libglib2.0-0 \
    libgl1-mesa-glx

# Copy your project files (assuming main.py and data.yaml)
COPY . .

# Step 4: Install YOLOv8 and other Python dependencies
RUN pip install --upgrade pip
#RUN pip install ultralytics 
RUN pip install -r requirements.txt

ENV MKL_THREADING_LAYER=GNU


# Optional: download pre-trained weights
#RUN yolo check && yolo export model=yolov8m.pt format=onnx

# Default command
#CMD ["python3", "bosch/scripts/main.py"]
