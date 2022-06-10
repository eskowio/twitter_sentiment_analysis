FROM python:3.8.5-slim

ENV TZ Europe/Warsaw

RUN adduser --disabled-password --gecos '' app && \
    mkdir /app && \
    chown app:app /app && \
    pip install --upgrade pip

USER app
WORKDIR /app

COPY --chown=app:app . .
RUN pip install --no-warn-script-location --user -r requirements.txt

ENV PATH="$HOME/.local/bin:${PATH}"

ENTRYPOINT ["/app/entrypoint"]
