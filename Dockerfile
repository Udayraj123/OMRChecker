ARG DEBIAN_VERSION=bookworm
ARG UV_VERSION=latest
ARG VARIANT=3.13


FROM ghcr.io/astral-sh/uv:$UV_VERSION AS uv


FROM python:$VARIANT-slim-$DEBIAN_VERSION
LABEL maintainer="udayraj123 <udayrajdeshmukh7+contact@gmail.com>"

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./

ENV PYTHONDONTWRITEBYTECODE=True
ENV PYTHONUNBUFFERED=True
ENV UV_LINK_MODE=copy

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    # For OpenCV etc...
    libgl1 libglib2.0-0 \
    # For numpy, etc
    gcc \
    # To remove the image size, it is recommended refresh the package cache as follows
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install project dependencies
RUN uv sync --frozen --no-install-project