CREATE ROLE twitt_producer LOGIN PASSWORD 'Loh6ziet';
CREATE ROLE superset LOGIN PASSWORD 'woh1xaeP';
CREATE DATABASE sentimental_analysis;

\c sentimental_analysis;

CREATE TABLE IF NOT EXISTS public.twitts (username VARCHAR(255) NOT NULL,
                                           text TEXT NOT NULL,
                                           created_at TIMESTAMP NOT NULL,
                                           sentiment VARCHAR(255) NOT NULL);

GRANT ALL PRIVILEGES ON DATABASE sentimental_analysis TO superset;
GRANT ALL PRIVILEGES ON DATABASE sentimental_analysis TO twitt_producer;

GRANT ALL PRIVILEGES ON TABLE public.twitts TO superset;
GRANT ALL PRIVILEGES ON TABLE public.twitts TO twitt_producer;