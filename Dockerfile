FROM python:3.8
COPY . /app
RUN apt-get update && yes | apt-get upgrade
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN export PYTHONPATH="${PYTHONPATH}:${PWD}"
CMD ["python3", "./app.py", "webcam"]
