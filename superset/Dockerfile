FROM apache/superset:1.3.0

USER root

RUN apt-get update && apt-get install -y \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip install mysqlclient \
    && pip install sqlalchemy-mongobi

COPY entrypoint /
RUN chmod a+x /entrypoint

USER superset

ENV LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN=1

ENTRYPOINT ["/entrypoint"]