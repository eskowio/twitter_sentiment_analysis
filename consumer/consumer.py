import os
import sys
import pika

from time import sleep

required_env = ['RABBITMQ_USER', 'RABBITMQ_PASSWORD', 'RABBITMQ_QUEUE','RABBITMQ_HOSTNAME']

f = open("/log/file.txt","w")

def callback(ch, method, properties, body):
    f.write(str(body,'utf-8') + "\n")
    f.flush()

def main():
    print("Reading configuration from environment variables...")
    config = {}
    for env in required_env:
        try:

            value = os.environ[env]
            print(env + ": " + value)
            config[env] = value
        except:
            print("Required {env} not found. Exiting.".format(env=env))
            sys.exit(255)

    credentials = pika.PlainCredentials(config['RABBITMQ_USER'], config['RABBITMQ_PASSWORD'])
    parameters = pika.ConnectionParameters(config['RABBITMQ_HOSTNAME'],5672,'/',credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)

    print("Ctrl+C to exit...")
    channel.basic_consume(queue=config['RABBITMQ_QUEUE'],auto_ack=True,on_message_callback=callback)
    channel.start_consuming()
    f.close()
main()
