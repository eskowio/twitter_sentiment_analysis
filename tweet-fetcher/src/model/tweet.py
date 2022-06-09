import json

import dateutil.parser as dp

class Tweet(object):
    def __init__(self, username, created_at, text):
        self._data = dict()
        self._data['user'] = username
        self._data['created_at'] = created_at
        self._data['text'] = text.encode().decode('utf-8')
        self._data['timestamp'] = int(dp.parse(created_at).timestamp())

    def to_json(self):
        return json.dumps(self._data, ensure_ascii=False)

    def from_json(self, payload):
        data = json.loads(payload)
        return Tweet(data['user'], data['created_at'], data['text'] )

    def get_username(self):
        return self._data['user']

    def get_created_at(self):
        return self._data['created_at']

    def get_text(self):
        return self._data['text']

    def get_timestamp(self):
        return self._data['timestamp']