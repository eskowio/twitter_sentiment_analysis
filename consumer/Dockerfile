FROM python:3

ADD consumer.py /
ENV http_proxy http://10.0.50.112:3128
ENV https_proxy http://10.0.50.112:3128

RUN mkdir -p /log && chmod 777 /log
RUN pip install pika

CMD [ "python", "./consumer.py" ]
