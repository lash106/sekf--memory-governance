# SEKF API. Build:  docker build -t sekf .     Run:  docker run --rm -p 8000:8000 sekf
# Defaults need no key and no network: SQLite store, rule-based Historian and Synthesizer,
# fake Judge. Configure with -e / --env-file (see .env.example); .env is never copied in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies pinned by constraints.txt (the same versions as `make setup`). The package is
# installed in editable mode on purpose: the app locates prompts/ and migrations relative to
# the source tree, which a copied-into-site-packages install would not have.
COPY pyproject.toml constraints.txt README.md ./
COPY src ./src
COPY prompts ./prompts
RUN pip install -c constraints.txt -e .

COPY datasets ./datasets
COPY scripts ./scripts

# Persistent data lives in one volume. The app runs as an unprivileged user.
RUN useradd --create-home --uid 10001 sekf \
    && mkdir -p /app/data \
    && chown -R sekf:sekf /app/data
ENV SEKF_DB_URL=sqlite:////app/data/sekf.sqlite3
VOLUME /app/data
USER sekf

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "sekf.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
