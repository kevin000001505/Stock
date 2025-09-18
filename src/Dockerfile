FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The first . mean the host machine -> The flder where the Dockerfile is located
# The second . means the container -> Here it mean /app because WORKDIR /app was set eariler
COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]