#!/usr/bin/env python3

import json

from lib.config import ConfigurationManager
from lib.logger import Logger
from lib.rabbitmq import RabbitWrapper
from lib.processor import TwitterSentimentalProcessor
from lib.postgresql import PostgreSQLWrapper
from model.tweet import Tweet

""" Application Main """

class App(object):   
    @staticmethod
    def main():

        # Check require properties
        require_props = ['RABBITMQ_USER', 
                         'RABBITMQ_PASSWORD', 
                         'RABBITMQ_QUEUE',
                         'RABBITMQ_HOSTNAME',
                         'RABBITMQ_PORT',
                         'POSTGRESQL_HOSTNAME',
                         'POSTGRESQL_PORT',
                         'POSTGRESQL_USERNAME',
                         'POSTGRESQL_PASSWORD',
                         'POSTGRESQL_DATABASE'
                        ]

        ConfigurationManager.checkRequireProperties(require_props)
        rabbitmq_props = ConfigurationManager.getRabbitMQProperties()
        db_props = ConfigurationManager.getPostgreSQLProperties()

        Logger.info("=== Data Processor ===")
        
        Logger.info("Initialize Sentimental Processor . . .")
        sentimental_processor = TwitterSentimentalProcessor(PostgreSQLWrapper(db_props))

        Logger.info("Initialize consumer . . .")
        consumer = RabbitWrapper(rabbitmq_props)
        consumer.consume(sentimental_processor.callback)

if __name__ == "__main__":
    App().main()