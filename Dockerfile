FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
RUN uv python install 3.13
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --python 3.13
