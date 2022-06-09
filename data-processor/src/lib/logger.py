from datetime import datetime

class Logger(object):
    @classmethod
    def log(cls,msg,severity):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("[{date}]     [{severity}]     {msg}".format(date=dt_string,severity=severity,msg=msg))

    @classmethod
    def info(cls,msg):
        cls.log(msg,"INFO")

    @classmethod
    def error(cls,msg):
        cls.log(msg,"ERROR")

    @classmethod
    def warning(cls,msg):
        cls.log(msg,"WARNING")
