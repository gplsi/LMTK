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
    --mock-modules)
      MOCK_MODULES=true
      shift
      ;;
    --help|-h)
      echo -e "Usage: $0 [options]"
      echo -e "Options:"
      echo -e "  --port, -p PORT   Specify the port to serve on (default: 8000)"
      echo -e "  --clean, -c       Clean existing build before building"
      echo -e "  --mock-modules    Mock problematic modules for documentation build"
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

# Create autodoc-mock file if mock flag is set
if [ "$MOCK_MODULES" = true ]; then
    echo -e "${YELLOW}Creating mock modules configuration for problematic imports...${NC}"
    cat > docs/source/conf_mock.py << 'EOF'
"""
Mock module configuration for documentation build.
This file is automatically imported by conf.py.
"""
import sys
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Mock modules that cause documentation build failures
MOCK_MODULES = [
    'wandb', 'wandb.sdk', 'wandb.sdk.data_types', 
    'numpy.float_', 'torch.cuda', 'torch.distributed',
    'torch.nn.parallel', 'torch.nn.parallel.distributed',
    'deepspeed'
]

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
EOF

    # Add import to conf.py if not already present
    if ! grep -q "import conf_mock" docs/source/conf.py; then
        echo -e "${YELLOW}Adding mock configuration import to conf.py...${NC}"
        sed -i '1s/^/# Import mock configuration\ntry:\n    import conf_mock\nexcept ImportError:\n    pass\n\n/' docs/source/conf.py
    fi
fi

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