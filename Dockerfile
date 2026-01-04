FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and watchlist
COPY sector_rotation.py .
COPY watchlist.csv .

# Set timezone for accurate market time detection
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Run the script
ENTRYPOINT ["python", "sector_rotation.py"]
