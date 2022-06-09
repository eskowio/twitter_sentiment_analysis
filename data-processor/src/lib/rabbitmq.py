import pika
from lib.logger import Logger 

class RabbitWrapper(object):
    
    def __init__(self, config):
        self._channel = pika.BlockingConnection(self._getConnectionParameters(config)).channel()
        Logger.info("RabbitMQ initialized successfully.")


    def _getConnectionParameters(self, config):
        self._queue = config['queue']
        credentials = pika.PlainCredentials(config['user'], config['password'])
        connectionParameters = pika.ConnectionParameters(config['hostname'],config['port'],'/',credentials)
        return connectionParameters
    
    def publish(self, message):
        Logger.info(f"Publishing message body: {message} to queue {self._queue}")
        self._channel.basic_publish(exchange='', routing_key=self._queue, body=message)

    def consume(self, callback):
        self._channel.basic_qos(prefetch_count=10)
        self._channel.basic_consume(queue=self._queue,auto_ack=True,on_message_callback=callback)
        self._channel.start_consuming()

class RabbitWrapperMock(object):
    
    def __init__(self):
        Logger.info("RabbitMQ Mock initialized successfully.")
    
    def publish(self, message):
        Logger.info(message)