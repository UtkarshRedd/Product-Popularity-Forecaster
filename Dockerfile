FROM python:3.8.6
RUN pip install --upgrade pip

# Create app directory
WORKDIR /app

# Install app dependencies
COPY requirements.txt ./

RUN pip install -r requirements.txt

# Bundle app source
COPY ./app /app
COPY app/config.py .
COPY app/models .
COPY app/flaskApp.py .
COPY app/data .
COPY app/templates /app/templates
COPY ./datasets /app/datasets

CMD [ "python3", "flaskApp.py" ]
