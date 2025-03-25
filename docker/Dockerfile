# Use official PyTorch image with pinned versions
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# Copy project files
COPY . /workspace/


# Then configure Git safety
RUN chown -R root:root /workspace && \
    git config --global --add safe.directory /workspace

# Verify configuration
RUN git config --global --list | grep safe.directory


# Configure entrypoint
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'Container ready - use exec to run commands' && tail -f /dev/null"]
