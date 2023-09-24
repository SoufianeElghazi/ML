# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application files to the container
COPY . /app

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which Streamlit runs (default is 8501)
EXPOSE 8501

# Command to run Streamlit app when the container starts
CMD ["streamlit", "run", "Interface.py"]
