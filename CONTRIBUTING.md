# Contributing to QCMet

[code-of-conduct]: CODE_OF_CONDUCT.md


Welcome! We're interested in contributions to Quantum Computing Metrics and Benchmarks. QCMet welcomes all community contributions from researchers, developers, and quantum computing enthusiasts. Please get in touch if you have any suggestions, comments or questions regarding the code or documentation. Unfortunately, we are unable to provide direct access to Issues and Merge Requests on the NPL Gitlab. As such, please feel free to reach out to [Deep Lall](mailto:deep.lall@npl.co.uk).

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Development Guidelines](#development-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to understand the standards we expect from our community.

## Getting Started

The following prerequsistes are required.
- **Python 3.10+**: Required for development
- **Git**: For version control
- **Basic Quantum Computing Knowledge**: Helpful but not required for all contributions

We first suggest reading the [README](README.md) and [documentation](https://qcmet.github.io/qcmet). Additionally, reading the accompanying paper ["A Review and Collection of Metrics and Benchmarks for Quantum Computers: definitions, methodologies and software"](https://doi.org/10.48550/arXiv.2502.06717) would be a good starting point.

## Ways to Contribute

If you have suggestions for how this project could be improved, or want to report a bug, please feel free to contact the development team.

For bug requests, it will be useful if you
1. **Provide detailed reproduction steps**
2. **Include system information** (OS, Python version, QCMet version)
3. **Share relevant code snippets or error messages**

If you have ideas for new features, it would be helpful if you  describe the use case and expected behavior.

### Documentation

This software is documented directly within the code through easy-to-follow docstrings for the different functional units of the code (in particular classes and functions).  When implementing new/changing  functionality, please add/update the docstrings accordingly and add further comments if necessary to follow the implementation.  Additionally, this software makes of the Sphinx for generation of user documentation.  In addition to providing a complete collection of the documentation for all functional units of the code, this documentation provides a central location for additional IPython notebook tutorial files describing usage of the code with explanation of the functionality and expected outcomes.  When implementing new functionality please consider whether it would be appropriate to add additional tutorials documenting the usage of this new functionality.


## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/qcmet/qcmet.git
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for development
pip install -e .[dev]

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### 3. Verify Installation

```bash
# Run tests to ensure everything works
pytest

# Run linter
ruff check .

# Build documentation
cd docs && make html
```

## Contribution Workflow

### 1. Create a Branch
Depending on your change, please use the following convention:

- `metric/your-metric-name`
- `feature/your-feature-name`
- `bugfix/your-bugfix-name`

```bash
# E.g. Create and switch to a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b bugfix/issue-number
```

### 2. Make Changes

- Follow our [development guidelines](#development-guidelines)
- Write or update tests for your changes
- Update documentation as needed
- Run tests locally before committing

### 3. Commit Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "Add: brief description of your changes

More detailed explanation of what was changed and why.
Fixes #issue-number (if applicable)"
```

### 4. Submit Pull/Merge Request

```bash
# Push your branch to your fork
git push origin feature/your-feature-name

# Create a pull request on GitHub
# Use the provided PR template
# Link to related issues
```

## Development Guidelines

### Code Style

The code in this repo follows that of  [Black's (stable) code style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html). It aims for "consistency, generality, readability and reducing git diffs".
- **Use Type Hints**: Add type annotations for function parameters and return values
- **Write Docstrings**: Document classes, methods, and functions
- **Keep Functions Small**: Aim for single-responsibility functions
- **Use Descriptive Names**: Variables and functions should be self-documenting

### Metric Implementation Guidelines

When implementing new quantum metrics:

1. **Inherit from BaseBenchmark**: Use the standard interface
2. **Document Methodology**: Include clear documentation of the metric methodology
3. **Specify Assumptions**: Document all assumptions and limitations
4. **Provide References**: Link to relevant academic papers or standards
5. **Include Examples**: Provide usage examples and expected outputs
6. **Validate Results**: Compare against known reference implementations


### Example Metric Template

```python
from qcmet.benchmarks import BaseBenchmark
from typing import Dict, Any, List

class YourMetric(BaseBenchmark):
    """
    Brief description of what this metric measures.

    This metric implements [Reference Paper/Standard].

    Assumptions:
    - List key assumptions
    - Include limitations

    Args:
        param1: Description of parameter
        param2: Description of parameter
    """

    def __init__(self, param1: int, param2: float, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2

    def _generate_circuits(self) -> None:
        """Generate quantum circuits for this metric."""
        # This should return either a list of qiskit.QuantumCircuit objects
        # or a list of dictionaries, where each dictionary has the key `circuit`
        # where the value is a qiskit.QuantumCircuit object. Other keys can
        # contain circuit related metadata which may be needed for analysis.
        pass

    def _analyze(self) -> Dict[str, Any]:
        """Analyze measurement results and compute metric value."""
        # Returns a dictionary of containing the result.
        pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/<path-to-your-test-file>

# Run with coverage
pytest --cov
```

### Writing Tests

- **Test Coverage**: Aim for >90% test coverage for new code
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test complete benchmark workflows


## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def your_function(param1: int, param2: str) -> Dict[str, Any]:
    """
    Brief description of the function.

    Longer description if needed, explaining the purpose,
    algorithm, or important details.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Dictionary containing result data with keys:
            - 'key1': Description of key1
            - 'key2': Description of key2

    Raises:
        ValueError: When param1 is negative.
        RuntimeError: When computation fails.

    Example:
        >>> result = your_function(42, "test")
        >>> print(result['key1'])
        42
    """
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e .[docs]

# Build documentation
cd docs
make html

# Open documentation
open _build/html/index.html  # macOS
# or
xdg-open _build/html/index.html  # Linux
```