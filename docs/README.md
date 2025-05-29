# ML Training Framework Documentation

This directory contains the Sphinx-based documentation for the ML Training Framework. The documentation is designed with a beautiful, responsive layout using the PyData Sphinx Theme with custom styling.

## Structure

- `source/`: Source files for the documentation
  - `conf.py`: Sphinx configuration
  - `index.rst`: Main index page
  - `_static/`: Static files (CSS, JS, images)
  - `_templates/`: Custom templates
  - `api/`: API reference documentation
  - `guides/`: User guides and tutorials
  - `examples/`: Example code and usage patterns

## Building the Documentation

We provide a convenient script to build the documentation:

```bash
# Build documentation
./build_docs.sh

# Build and open in browser
./build_docs.sh --open

# Check links in documentation
./build_docs.sh --linkcheck

# Run doctests
./build_docs.sh --doctest

# Clean and rebuild
./build_docs.sh --clean
```

Alternatively, you can use tox to build the documentation:

```bash
# Using tox
tox -e docs
```

## Documentation Features

Our documentation includes:

- **Beautiful Design**: Custom color palette with gradient headers and responsive layout
- **Interactive Elements**: Copyable code blocks, animated feature boxes, and smooth navigation
- **API Auto-Documentation**: Automatic extraction of docstrings from the codebase
- **Mermaid Diagrams**: Flow charts and diagrams for visual explanation
- **Syntax Highlighting**: Beautiful code highlighting with line numbers
- **Dark Mode Support**: Seamless switching between light and dark modes
- **Responsive Layout**: Works on all device sizes from mobile to desktop

## Contributing to Documentation

When adding new documentation, please follow these guidelines:

1. Use reStructuredText (.rst) for most documentation
2. Follow the existing style patterns
3. Include code examples where appropriate
4. Use admonitions (note, warning, tip) to highlight important information
5. Add any new pages to the appropriate toctree in index.rst
6. Test your documentation with `./build_docs.sh --doctest`

## Color Palette

The documentation uses a custom color palette defined in `_static/css/custom.css`:

- Primary: `#4C72B0` - Used for main elements and headers
- Secondary: `#7B64FF` - Used for gradients and accents
- Accent: `#FF7F0E` - Used for highlights and interactive elements
- Success: `#55A868` - Used for positive elements
- Warning: `#FDB462` - Used for warning elements
- Danger: `#D65F5F` - Used for error elements and critical warnings

## Continuous Integration

Documentation is automatically built and tested in CI through our GitHub Actions workflow. Any failure in documentation building will block the CI pipeline.

## Hosting

The documentation is hosted on GitHub Pages and is automatically deployed when commits are pushed to the main branch.