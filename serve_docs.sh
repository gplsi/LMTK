#!/bin/bash
# serve_docs.sh - Script to build and serve documentation on a local HTTP server

set -e

# Set colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default port
PORT=8000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port|-p)
      PORT="$2"
      shift
      shift
      ;;
    --clean|-c)
      echo -e "${YELLOW}Cleaning existing documentation build...${NC}"
      rm -rf docs/build
      shift
      ;;
    --help|-h)
      echo -e "Usage: $0 [options]"
      echo -e "Options:"
      echo -e "  --port, -p PORT   Specify the port to serve on (default: 8000)"
      echo -e "  --clean, -c       Clean existing build before building"
      echo -e "  --help, -h        Show this help message"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo -e "Use --help to see available options"
      exit 1
      ;;
  esac
done

# Create build directory if it doesn't exist
mkdir -p docs/build

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Error: poetry is not installed. Please install it first.${NC}"
    exit 1
fi

# Check that all necessary directories exist
if [ ! -d "docs/source" ]; then
    echo -e "${RED}Error: docs/source directory not found.${NC}"
    exit 1
fi

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
poetry install

# Build HTML documentation
echo -e "${YELLOW}Building HTML documentation...${NC}"
poetry run sphinx-build -b html docs/source docs/build/html

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}HTML documentation built successfully!${NC}"
else
    echo -e "${RED}Error building HTML documentation.${NC}"
    exit 1
fi

# Find an available port if the default is in use
if ! nc -z localhost $PORT >/dev/null 2>&1; then
    echo -e "${BLUE}Using port $PORT for the documentation server${NC}"
else
    # Try to find an available port starting from the specified one
    ORIGINAL_PORT=$PORT
    for (( i=0; i<100; i++ )); do
        PORT=$((ORIGINAL_PORT + i))
        if ! nc -z localhost $PORT >/dev/null 2>&1; then
            echo -e "${YELLOW}Port $ORIGINAL_PORT is in use. Using port $PORT instead.${NC}"
            break
        fi
    done
fi

# Serve documentation
echo -e "${GREEN}Starting documentation server on port $PORT...${NC}"
echo -e "${BLUE}You can access the documentation at: http://localhost:$PORT${NC}"
echo -e "${YELLOW}Opening documentation in your browser...${NC}"

# Open browser
"$BROWSER" "http://localhost:$PORT" &

# Start the server
cd docs/build/html && python -m http.server $PORT

# This line will only be reached when the server is stopped
echo -e "${GREEN}Documentation server stopped.${NC}"