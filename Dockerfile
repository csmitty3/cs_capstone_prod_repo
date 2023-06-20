FROM python:3.9

# set the working directory
WORKDIR /app

# install dependencies
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5000

# start the server
CMD python ./main.py