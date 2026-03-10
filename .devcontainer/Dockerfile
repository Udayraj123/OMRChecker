ARG UV_VERSION=latest
ARG PYTHON_VERSION=3.11-bullseye


FROM ghcr.io/astral-sh/uv:$UV_VERSION AS uv


FROM mcr.microsoft.com/vscode/devcontainers/python:$PYTHON_VERSION
LABEL maintainer="udayraj123 <udayrajdeshmukh7+contact@gmail.com>"

# hadolint ignore=DL3008
RUN export DEBIAN_FRONTEND=noninteractive \
    export OMR_CHECKER_CONTAINER=true \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    # For OpenCV etc...
    libgl1 libglib2.0-0 \
    # For numpy, etc
    gcc \
    # To remove the image size, it is recommended refresh the package cache as follows
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv --chown=vscode /uv /uvx /bin/

# Note: "postCreateCommand" and "postStartCommand" handle this
# RUN uv sync --frozen --no-install-project
# ENTRYPOINT ["./run-devcontainer.sh"]
