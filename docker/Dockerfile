# Use official PyTorch image with pinned versions
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    POETRY_VERSION=2.1.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Create workspace directory with proper permissions
RUN mkdir -p /workspace && chmod 1777 /workspace
WORKDIR /workspace

RUN echo 'export PYTHONPATH=/workspace/src:$PYTHONPATH' > /etc/profile.d/pythonpath.sh

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    chmod 755 $POETRY_HOME/bin/poetry

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies (including development dependencies)
RUN poetry install --no-root

# Copy project files
COPY . /workspace/

# Install project with all dependencies
RUN poetry install

# Configure entrypoint
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'Container ready - use exec to run commands' && tail -f /dev/null"]
