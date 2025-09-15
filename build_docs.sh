#!/bin/bash
# build_docs.sh - Script to build and view documentation

set -e

# Set colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}   ML Training Framework Documentation Builder ${NC}"
echo -e "${BLUE}===============================================${NC}"

# Check for command line arguments
BUILD_TYPE="html"
OPEN_BROWSER=false
CHECK_LINKS=false
RUN_DOCTESTS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --open|-o)
      OPEN_BROWSER=true
      shift
      ;;
    --linkcheck|-l)
      CHECK_LINKS=true
      shift
      ;;
    --doctest|-d)
      RUN_DOCTESTS=true
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
      echo -e "  --open, -o        Open documentation in browser after build"
      echo -e "  --linkcheck, -l   Check external links in documentation"
      echo -e "  --doctest, -d     Run doctests in the documentation"
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
    echo -e "Documentation available at: ${BLUE}docs/build/html/index.html${NC}"
else
    echo -e "${RED}Error building HTML documentation.${NC}"
    exit 1
fi

# Run link check if requested
if [ "$CHECK_LINKS" = true ]; then
    echo -e "${YELLOW}Checking external links...${NC}"
    poetry run sphinx-build -b linkcheck docs/source docs/build/linkcheck
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Link check completed successfully!${NC}"
    else
        echo -e "${RED}Link check found issues. See docs/build/linkcheck/output.txt for details.${NC}"
    fi
fi

# Run doctests if requested
if [ "$RUN_DOCTESTS" = true ]; then
    echo -e "${YELLOW}Running doctests...${NC}"
    poetry run sphinx-build -b doctest docs/source docs/build/doctest
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Doctests completed successfully!${NC}"
    else
        echo -e "${RED}Doctests found issues. See output above for details.${NC}"
    fi
fi

# Open in browser if requested
if [ "$OPEN_BROWSER" = true ]; then
    echo -e "${YELLOW}Opening documentation in browser...${NC}"
    DOC_PATH="$(pwd)/docs/build/html/index.html"
    
    if command -v $BROWSER &> /dev/null; then
        $BROWSER "$DOC_PATH"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "$DOC_PATH"
    elif command -v open &> /dev/null; then
        open "$DOC_PATH"
    else
        echo -e "${RED}Could not open browser automatically.${NC}"
        echo -e "Please open ${BLUE}$DOC_PATH${NC} manually."
    fi
fi

echo -e "${GREEN}Documentation process completed!${NC}"