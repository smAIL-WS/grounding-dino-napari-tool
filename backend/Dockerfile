# Replace this with the actual name/tag of your current Docker image
FROM hswt555har/mmdetection-models:v1.1

# Install the API dependencies
RUN pip install fastapi uvicorn python-multipart

# Copy the server script and config into the container
WORKDIR /app
COPY server.py /app/server.py
COPY config.py /app/config.py

# Tell the container to start the API server
CMD ["python", "server.py"]