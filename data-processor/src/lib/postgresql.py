import psycopg2

from lib.logger import Logger 

class PostgreSQLWrapper(object):
    
    def __init__(self, config):
        self._config = config

    def insert_twitt(self, username, text, created_at, sentiment):
        query =  "INSERT INTO twitts (username, text, created_at, sentiment) VALUES (%s, %s, %s, %s);"

        connection = psycopg2.connect(user=self._config['username'],
                                            password=self._config['password'],
                                            host=self._config['hostname'],
                                            port=self._config['port'],
                                            database=self._config['database'])
        
        cursor = connection.cursor()

        data = (username, text, created_at, sentiment)

        cursor.execute(query, data)

        connection.commit()
        