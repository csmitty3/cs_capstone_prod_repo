FROM python:3.9
WORKDIR /app
#COPY requirements.txt .
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5000
# start the server

CMD ["python", "main.py"]