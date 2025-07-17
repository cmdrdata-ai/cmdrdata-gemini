# Contributing to cmdrdata-openai

Thank you for your interest in contributing to cmdrdata-openai! This guide will help you get started.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cmdrdata-openai.git
   cd cmdrdata-openai
   ```
3. **Install development dependencies**:
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install project with dev dependencies
   uv pip install -e .[dev]
   ```
4. **Install pre-commit hooks**:
   ```bash
   uv run pre-commit install
   ```

## 🧪 Testing

We maintain **100% test pass rate**. All contributions must include tests.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=cmdrdata_openai --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_client.py -v
```

### Writing Tests

- Place test files in the `tests/` directory
- Follow the naming convention `test_*.py`
- Use descriptive test method names
- Include both positive and negative test cases
- Mock external dependencies (OpenAI API, HTTP requests)

Example test structure:
```python
def test_feature_success(self):
    """Test successful feature operation"""
    # Arrange
    # Act
    # Assert

def test_feature_failure(self):
    """Test feature failure handling"""
    # Test error conditions
```

## 🎨 Code Style

We use automated formatting and linting:

```bash
# Format code
uv run black cmdrdata_openai/

# Sort imports
uv run isort cmdrdata_openai/

# Type checking
uv run mypy cmdrdata_openai/ --ignore-missing-imports

# Security check
uv run safety check
```

### Style Guidelines

- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking
- **pytest** for testing
- Follow PEP 8 and PEP 257
- Use type hints for all public APIs
- Include docstrings for all public functions/classes

## 🔧 Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code
   - Add tests
   - Update documentation if needed

3. **Run quality checks**:
   ```bash
   # This runs automatically with pre-commit
   uv run black cmdrdata_openai/
   uv run isort cmdrdata_openai/
   uv run mypy cmdrdata_openai/
   uv run pytest
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub

## 📝 Commit Message Convention

We follow conventional commits:

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `test:` adding or updating tests
- `refactor:` code refactoring
- `perf:` performance improvements
- `chore:` maintenance tasks

Examples:
```
feat: add async support for usage tracking
fix: handle network timeouts gracefully
docs: update README with new examples
test: add tests for edge cases
```

## 🐛 Bug Reports

When reporting bugs, please include:

- Python version
- OpenAI SDK version
- cmdrdata-openai version
- Minimal code example
- Error traceback
- Expected vs actual behavior

## 💡 Feature Requests

For new features:

- Check existing issues first
- Describe the use case
- Provide implementation ideas
- Consider backward compatibility

## 🔒 Security

- Never commit API keys or secrets
- Report security issues privately to hello@cmdrdata.ai
- Use secure coding practices
- Validate all inputs
- Sanitize logs and outputs

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass (100% pass rate required)
- [ ] New features include tests
- [ ] Documentation updated if needed
- [ ] Type hints added for new code
- [ ] No breaking changes (unless discussed)
- [ ] Commit messages follow convention
- [ ] Security considerations addressed

## 🔄 CI/CD

Our GitHub Actions will automatically:

- Run tests across Python 3.8-3.12
- Check code formatting and style
- Run type checking
- Generate coverage reports
- Check for security vulnerabilities

All checks must pass before merging.

## 📞 Getting Help

- **Discord**: [Join our community](https://discord.gg/cmdrdata)
- **Email**: hello@cmdrdata.ai
- **Issues**: [GitHub Issues](https://github.com/cmdrdata-ai/cmdrdata-openai/issues)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for making cmdrdata-openai better! 🙏
