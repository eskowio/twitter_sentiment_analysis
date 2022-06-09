import json

from tweepy import StreamingClient, StreamRule

from model.tweet import  Tweet
from lib.logger import Logger

class TwitterFetcher(object):
    def __init__(self, config, producer):
        self._config = config
        self._producer = producer

    def start(self, topic):
        _stream = TwitterStreamListener(bearer_token=self._config['bearer.token'],
                                        producer=self._producer)
        
        Logger.info("Initializing Stream.")

        rules = _stream.get_rules().data
        Logger.info("Cleaning existing rules.")
        if rules:
            Logger.info(f"Rules to delete: {rules}")
            for rule in rules:
                _stream.delete_rules([rule.id])
                Logger.info(f"Deleteing rule: {rule.id}")

        _stream.add_rules(
            [
                StreamRule(value=topic)
            ]
        )

        Logger.info(f"Loaded rules: {_stream.get_rules().data}")

        Logger.info("Start streaming Twitts.")
        _stream.filter(user_fields=['username'], tweet_fields=['created_at'], expansions=['author_id'])

class TwitterStreamListener(StreamingClient):
    def __init__(self, bearer_token, producer):
        
        StreamingClient.__init__(self, bearer_token=bearer_token)
        self._producer = producer
    
    def on_data(self, raw_data):

        data = json.loads(raw_data)
        payload = Tweet(data['includes']['users'][0]['username'], 
                        data['data']['created_at'],
                        data['data']['text'])

        Logger.info(f"Fetched twitt: {payload.to_json()}")
        self._producer.publish(payload.to_json())
    
        return True

    def on_error(self, status):
        if status == 420:
            return False
        print(status)