FROM python:3.11-slim
WORKDIR /app

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY ..
EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port", "8080"]