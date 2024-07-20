FROM umihico/aws-lambda-selenium-python:latest

COPY requirements.txt ./

RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf && pip install -r requirements.txt -t .

COPY src/ ./src/

COPY app.py ./

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
