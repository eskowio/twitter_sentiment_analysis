#!/usr/bin/env python3

from lib.config import ConfigurationManager
from lib.logger import Logger
from lib.rabbitmq import RabbitWrapper, RabbitWrapperMock
from lib.twitter import TwitterFetcher

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
                         'TWITTER_BEARER_TOKEN',
                         'TWITTER_TOPIC',
                         'RABBITMQ_MOCK'
                        ]

        ConfigurationManager.checkRequireProperties(require_props)
        rabbitmq_props = ConfigurationManager.getRabbitMQProperties()
        twitter_props = ConfigurationManager.getTwitterProperties()

        Logger.info("=== Twitter Fetcher ===")
        Logger.info(f"Configured topic: {twitter_props['topic']}")

        Logger.info("Initialize producer . . .")
        producer = None
        if rabbitmq_props['mock'] == "false":
            producer = RabbitWrapper(rabbitmq_props)
        else:
            producer = RabbitWrapperMock()


        fetcher = TwitterFetcher(twitter_props, producer)

        fetcher.start(twitter_props['topic'])

if __name__ == "__main__":
    App().main()