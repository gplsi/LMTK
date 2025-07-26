# Use official PyTorch image with pinned versions
FROM nvcr.io/nvidia/pytorch:25.02-py3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Arguments for user creation (will be passed from docker build)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=lmtk_user

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
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create user with matching UID/GID from host
RUN if getent group $GROUP_ID >/dev/null; then \
      echo "Group with GID $GROUP_ID already exists."; \
    else \
      groupadd -g $GROUP_ID $USERNAME; \
    fi && \
    if id -u $USER_ID >/dev/null 2>&1; then \
      echo "User with UID $USER_ID already exists."; \
    else \
      useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash $USERNAME; \
    fi && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Ensure all files in /workspace are owned by the mapped user (handles .git and all contents)
RUN chown -R $USER_ID:$GROUP_ID /workspace

# Setup poetry configuration
RUN pip install -U pip setuptools
RUN pip install poetry==1.8.4
RUN poetry config virtualenvs.create false

# Copy Poetry configuration files first for better caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies only (no dev dependencies for production)
RUN poetry install --only=main --no-root && \
    rm -rf $POETRY_CACHE_DIR

# Copy project files
COPY . /workspace/

# Install the project itself
RUN poetry install --only-root

# Set proper ownership of workspace
RUN chmod -R 770 /workspace

# Switch to the created user
USER $USERNAME

# Configure entrypoint
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'Container ready - use exec to run commands' && tail -f /dev/null"]
