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
RUN set -e && \
    # First check if group exists, if not create it
    if ! getent group $GROUP_ID >/dev/null; then \
        groupadd -g $GROUP_ID $USERNAME; \
    fi && \
    # Get the group name for the GID (might be different from USERNAME)
    GROUP_NAME=$(getent group $GROUP_ID | cut -d: -f1) && \
    # Remove any existing user with this UID to avoid conflicts
    if id -u $USER_ID >/dev/null 2>&1; then \
        userdel -f $(id -nu $USER_ID) 2>/dev/null || true; \
    fi && \
    # Create the user with specified UID and primary group
    useradd -u $USER_ID -g $GROUP_NAME -m -s /bin/bash $USERNAME && \
    # Add to sudoers
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    # Verify user creation
    id $USERNAME && \
    echo "User $USERNAME created successfully with UID $USER_ID and GID $GROUP_ID"

# Add a more robust NSS configuration and fallback
RUN echo "passwd: files" > /etc/nsswitch.conf.new && \
    echo "group: files" >> /etc/nsswitch.conf.new && \
    echo "shadow: files" >> /etc/nsswitch.conf.new && \
    echo "hosts: files dns" >> /etc/nsswitch.conf.new && \
    mv /etc/nsswitch.conf.new /etc/nsswitch.conf

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
