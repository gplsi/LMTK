# Use official PyTorch image with pinned versions
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/workspace/src:$PYTHONPATH

# Create workspace directory with proper permissions
RUN mkdir -p /workspace && chmod 1777 /workspace
WORKDIR /workspace

# Add PYTHONPATH to both bash and profile config (ensures it works in all scenarios)
RUN echo 'export PYTHONPATH=/workspace/src:$PYTHONPATH' >> /etc/bash.bashrc && \
    echo 'export PYTHONPATH=/workspace/src:$PYTHONPATH' > /etc/profile.d/pythonpath.sh && \
    chmod +x /etc/profile.d/pythonpath.sh

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

# Create a template .env file for wandb API key
RUN echo '# WandB API Key - Replace with your key or use environment variable' > /workspace/.env && \
    echo 'WANDB_API_KEY=${WANDB_API_KEY}' >> /workspace/.env && \
    chmod 600 /workspace/.env

# Then configure Git safety
RUN chown -R root:root /workspace && \
    git config --global --add safe.directory /workspace

# Verify configuration
RUN git config --global --list | grep safe.directory

# Add a helper script to set up wandb API key
RUN echo '#!/bin/bash\n\
if [ -n "$1" ]; then\n\
  echo "WANDB_API_KEY=$1" > /workspace/.env\n\
  echo "WandB API key has been set"\n\
else\n\
  echo "Usage: setup-wandb <api_key>"\n\
fi\n' > /usr/local/bin/setup-wandb && \
    chmod +x /usr/local/bin/setup-wandb

# Configure entrypoint
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'Container ready - use exec to run commands'; echo 'To set WandB API key, run: setup-wandb YOUR_API_KEY'; tail -f /dev/null"]
