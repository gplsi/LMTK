# Docker Image Issues Troubleshooting

## Problem Description

If you see this error in your SLURM job output:

```
Unable to find image 'lmtk:latest' locally
docker: Error response from daemon: pull access denied for lmtk, repository does not exist or may require 'docker login': denied: requested access to the resource is denied
```

But the logs show "✅ Using existing Docker image: lmtk:latest", this indicates a Docker image corruption or cache issue.

## Root Cause

This happens when:
1. `docker image inspect` returns success (image metadata exists) 
2. But `docker run` fails because the actual image layers are missing or corrupted
3. Docker tries to pull from Docker Hub, but `lmtk:latest` is a local image that doesn't exist remotely

## Solutions

### Option 1: Force Rebuild (Recommended)
```bash
./submit_job.sh -c config/experiments/your_config.yaml --rebuild
```

The `--rebuild` flag will:
- Force remove any existing image
- Rebuild the Docker image from scratch
- Skip the image existence check

### Option 2: Manual Docker Cleanup
```bash
# Remove the corrupted image
docker rmi lmtk:latest

# Then submit normally
./submit_job.sh -c config/experiments/your_config.yaml
```

### Option 3: Build Image Manually
```bash
# Build the image manually first
docker build \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    --build-arg USERNAME=$(whoami) \
    --network=host \
    -t lmtk:latest \
    -f docker/Dockerfile \
    .

# Then submit normally
./submit_job.sh -c config/experiments/your_config.yaml
```

## Prevention

To avoid this issue:
- Use `--rebuild` after making code changes
- Regularly clean up Docker cache: `docker system prune`
- Monitor Docker disk usage: `docker system df`

## Technical Details

The fix improves the Docker image validation by:
1. Using `docker run --rm --entrypoint /bin/echo` to actually test the image
2. Adding better error handling and debugging information
3. Providing a force rebuild option to skip checks entirely
4. Adding cleanup steps to remove corrupted images before rebuilding

## Updated Workflow

The improved workflow now:
1. Checks if force rebuild is requested → rebuilds immediately
2. Otherwise, tests image with a simple command
3. If test fails → removes corrupted image and rebuilds
4. Adds detailed debugging output on failures
