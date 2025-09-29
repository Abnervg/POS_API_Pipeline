# 1. Start with an official Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy and install dependencies first to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your entire project into the container's working directory
COPY . .

# 5. Define the default command to run when the container starts
CMD ["python", "main.py", "--step", "daily_run"]

