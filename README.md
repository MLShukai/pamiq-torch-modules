# P-AMI\<Q> Torch Modules

A collection of custom PyTorch modules designed for AMI research, with a focus on robust implementation, type safety, and thorough testing.

## Features

- **Well-tested components**: Comprehensive test suite ensuring reliability
- **Type annotations**: Complete type hints for better IDE support and code quality
- **PyTorch integration**: Seamless integration with PyTorch ecosystem
- **Cross-platform**: Tested on Linux, macOS, and Windows
- **Python 3.10+**: Leveraging modern Python features

## Installation

```bash
# Using pip
pip install https://github.com/MLShukai/pamiq-torch-modules

# Using uv
uv add https://github.com/MLShukai/pamiq-torch-modules
```

## Available Modules

The library provides several specialized PyTorch modules:

- **ResNet**: Feed-forward residual networks
- **SioConv**: State-space model with Input-Output Convolution
- **OneHot**: Tools for handling one-hot encoded data
- **Normal Distributions**: Customizable wrappers for normal distributions
- **Categorical Distributions**: Multi-categorical distribution implementations
- **JEPA**: Joint Embedding Predictive Architecture implementations

## Usage Examples

```python
import torch
from pamiq_torch_modules.models import resnet, sioconv, one_hot

# Create a residual network feed-forward module
model = resnet.ResNetFF(dim=256, dim_hidden=512, depth=4)
x = torch.randn(32, 256)
output = model(x)  # Shape: [32, 256]

# Use SioConv for sequence processing
batch_size, seq_len, dim = 16, 10, 64
model = sioconv.SioConvPS(depth=3, dim=dim, dim_ff_hidden=128)
x = torch.randn(batch_size, seq_len, dim)
hidden = torch.randn(batch_size, 3, dim)
output, new_hidden = model(x, hidden)
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/MLShukai/pamiq-torch-modules
cd pamiq-torch-modules

# Set up the environment (using uv)
uv sync
source .venv/bin/activate  # On Linux/macOS
.venv\Scripts\activate  # On Windows

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
make test

# Type checking
make type

# Format code
make format

# Run the full development workflow
make run
```

## License

[MIT License](LICENSE)
