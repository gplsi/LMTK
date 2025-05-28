# Use official PyTorch image with pinned versions
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=1 \
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
    python3.10 \
    python3.10-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    chmod 755 $POETRY_HOME/bin/poetry

# Set up Poetry to use in-project virtualenvs
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true

# Copy project files
COPY . /workspace/
WORKDIR /workspace

# Create a script to initialize the environment
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Activate virtual environment if it exists' >> /entrypoint.sh && \
    echo 'if [ -f ".venv/bin/activate" ]; then' >> /entrypoint.sh && \
    echo '    source .venv/bin/activate' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# If no command is provided, start a shell' >> /entrypoint.sh && \
    echo 'if [ "$#" -eq 0 ]; then' >> /entrypoint.sh && \
    echo '    exec /bin/bash' >> /entrypoint.sh && \
    echo 'else' >> /entrypoint.sh && \
    echo '    exec "$@"' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set up entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
