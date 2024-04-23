# Use the official Python image as base image
FROM python:3.12.1-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
RUN pip install --upgrade pip

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
# Run app.py when the container launches
CMD ["python", "app.py"]