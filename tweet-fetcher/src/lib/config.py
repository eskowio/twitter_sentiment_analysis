import os
import sys

from lib.logger import Logger

class ConfigurationManager(object):
    @classmethod
    def getProperty(cls,property_name):
        return os.environ.get(property_name)
    
    @classmethod
    def getPrefixProperties(cls, prefix):
        prefix_properties = {}
        for key, value in os.environ.items():
            env_name = str(key.lower())
            if env_name.startswith(prefix[:-1]):
                env_name = env_name.replace(prefix, "")
                env_name = env_name.replace("_", ".")
                prefix_properties[env_name] = value
        
        return prefix_properties

    @classmethod
    def getRabbitMQProperties(cls):
        return cls.getPrefixProperties("rabbitmq_")

    @classmethod
    def getTwitterProperties(cls):
        return cls.getPrefixProperties("twitter_")

    @classmethod
    def requireProperty(cls,property_name):
        p = cls.getProperty(property_name)
        if p is None:
            Logger.error(f"Required property {property_name} is not defined. Exiting.")
            sys.exit(255)
        else:
            return p

    @classmethod
    def checkRequireProperties(cls,property_names):
        for property_name in property_names: cls.requireProperty(property_name)