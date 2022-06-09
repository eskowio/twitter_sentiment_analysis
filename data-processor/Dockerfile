FROM python:3.8.5-slim

ENV TZ Europe/Warsaw

RUN adduser --disabled-password --gecos '' app && \
    mkdir /app && \
    chown app:app /app && \
    pip install --upgrade pip

USER app
WORKDIR /app

COPY --chown=app:app /src/requirements.txt .
RUN pip install --no-warn-script-location --user -r requirements.txt

COPY --chown=app:app /src .

ENV PATH="$HOME/.local/bin:${PATH}"

ENTRYPOINT ["/app/entrypoint"]