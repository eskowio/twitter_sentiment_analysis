import pika
from lib.logger import Logger 

class RabbitProducer(object):
    
    def __init__(self, config):
        self._channel = pika.BlockingConnection(self._getConnectionParameters(config)).channel()
        Logger.info("RabbitMQ producer initialized successfully.")


    def _getConnectionParameters(self, config):
        self._queue = config['queue']
        credentials = pika.PlainCredentials(config['user'], config['password'])
        connectionParameters = pika.ConnectionParameters(config['hostname'],config['port'],'/',credentials)
        return connectionParameters
    
    def publish(self, message):
        Logger.info(f"Publishing message body: {message} to queue {self._queue}")
        self._channel.basic_publish(exchange='', routing_key=self._queue, body=message)

class RabbitProducerMock(object):
    
    def __init__(self):
        Logger.info("RabbitMQ Mock initialized successfully.")
    
    def publish(self, message):
        Logger.info(message)