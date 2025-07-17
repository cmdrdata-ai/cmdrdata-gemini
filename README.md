# cmdrdata-gemini

[![CI](https://github.com/cmdrdata-ai/cmdrdata-gemini/workflows/CI/badge.svg)](https://github.com/cmdrdata-ai/cmdrdata-gemini/actions)
[![codecov](https://codecov.io/gh/cmdrdata-ai/cmdrdata-gemini/branch/main/graph/badge.svg)](https://codecov.io/gh/cmdrdata-ai/cmdrdata-gemini)
[![PyPI version](https://badge.fury.io/py/cmdrdata-gemini.svg)](https://badge.fury.io/py/cmdrdata-gemini)
[![Python Support](https://img.shields.io/pypi/pyversions/cmdrdata-gemini.svg)](https://pypi.org/project/cmdrdata-gemini/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Transparent usage tracking for Google Gemini API calls**

cmdrdata-gemini provides drop-in replacements for Google's Gen AI Python SDK clients that automatically track usage for customer billing and analytics without requiring any changes to your existing code.

## 🛡️ Production Ready

**Extremely robust and reliable** - Built for production environments with:

- **100% Test Coverage** - Comprehensive tests ensuring reliability
- **Non-blocking I/O** - Fire-and-forget tracking never slows your app
- **Zero Code Changes** - Drop-in replacement for existing Google Gen AI clients
- **Thread-safe** - Safe for concurrent applications
- **Error Resilient** - Your app continues even if tracking fails

## 🚀 Quick Start

### Installation

```bash
pip install cmdrdata-gemini
```

### Basic Usage

```python
# Before
from google import genai
client = genai.Client(api_key="your-gemini-key")

# After - same API, automatic tracking!
import cmdrdata_gemini
client = cmdrdata_gemini.TrackedGemini(
    api_key="your-gemini-key",
    cmdrdata_api_key="your-cmdrdata-key"
)

# Same API as regular Google Gen AI client
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works"
)

print(response.text)
# Usage automatically tracked to cmdrdata backend!
```

### Async Support

```python
import cmdrdata_gemini

async def main():
    client = cmdrdata_gemini.AsyncTrackedGemini(
        api_key="your-gemini-key", 
        cmdrdata_api_key="your-cmdrdata-key"
    )
    
    response = await client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello, Gemini!"
    )
    
    print(response.text)
    # Async usage tracking included!
```

## 🎯 Customer Context Management

### Automatic Customer Tracking

```python
from cmdrdata_gemini.context import customer_context

# Set customer context for automatic tracking
with customer_context("customer-123"):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Help me code"
    )
    # Automatically tracked for customer-123!

# Or pass customer_id directly
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello",
    customer_id="customer-456"  # Direct customer ID
)
```

### Manual Context Management

```python
from cmdrdata_gemini.context import set_customer_context, clear_customer_context

# Set context for current thread
set_customer_context("customer-789")

response = client.models.generate_content(...)  # Tracked for customer-789

# Clear context
clear_customer_context()
```

## ⚙️ Configuration

### Environment Variables

```bash
# Optional: Set via environment variables
export GEMINI_API_KEY="your-gemini-key"
export CMDRDATA_API_KEY="your-cmdrdata-key"
export CMDRDATA_ENDPOINT="https://api.cmdrdata.ai/events"  # Optional
```

```python
# Then use without passing keys
client = cmdrdata_gemini.TrackedGemini()
```

### Custom Configuration

```python
client = cmdrdata_gemini.TrackedGemini(
    api_key="your-gemini-key",
    cmdrdata_api_key="your-cmdrdata-key",
    cmdrdata_endpoint="https://your-custom-endpoint.com/events",
    track_usage=True,  # Enable/disable tracking
    timeout=30,  # Custom timeout
    max_retries=3  # Custom retry logic
)
```

## 🔒 Security & Privacy

### Automatic Data Sanitization

- **API keys automatically redacted** from logs
- **Sensitive data sanitized** before transmission
- **Input validation** prevents injection attacks
- **Secure defaults** for all configuration

### What Gets Tracked

```python
# Tracked data (anonymized):
{
    "customer_id": "customer-123",
    "model": "gemini-2.5-flash", 
    "input_tokens": 25,
    "output_tokens": 150,
    "total_tokens": 175,
    "provider": "google",
    "timestamp": "2025-01-15T10:30:00Z",
    "metadata": {
        "response_id": "resp_abc123",
        "model_version": "001",
        "finish_reason": "STOP",
        "safety_ratings": null
    }
}
```

**Note**: Message content is never tracked - only metadata and token counts.

## 📊 Monitoring & Performance

### Built-in Performance Monitoring

```python
# Get performance statistics
stats = client.get_performance_stats()
print(f"Average response time: {stats['api_calls']['avg']}ms")
print(f"Total API calls: {stats['api_calls']['count']}")
```

### Health Monitoring

```python
# Check tracking system health
tracker = client.get_usage_tracker()
health = tracker.get_health_status()
print(f"Tracking healthy: {health['healthy']}")
```

## 🛠️ Advanced Usage

### Token Counting

```python
# Count tokens without generating content (also tracked)
token_count = client.models.count_tokens(
    model="gemini-2.5-flash",
    contents="How many tokens is this?"
)
print(f"Token count: {token_count.total_tokens}")
```

### Disable Tracking for Specific Calls

```python
# Disable tracking for sensitive operations
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Private query",
    track_usage=False  # This call won't be tracked
)
```

### Error Handling

```python
from cmdrdata_gemini.exceptions import CMDRDataError, TrackingError

try:
    client = cmdrdata_gemini.TrackedGemini(
        api_key="invalid-key",
        cmdrdata_api_key="invalid-cmdrdata-key"
    )
except CMDRDataError as e:
    print(f"Configuration error: {e}")
    # Handle configuration issues
```

### Integration with Existing Error Handling

```python
# All original Google Gen AI exceptions work the same way
try:
    response = client.models.generate_content(...)
except Exception as e:  # Google Gen AI exceptions
    print(f"Google Gen AI error: {e}")
    # Your existing error handling works unchanged
```

## 🔧 Development

### Requirements

- Python 3.9+
- google-genai>=0.1.0

### Installation for Development

```bash
git clone https://github.com/cmdrdata-ai/cmdrdata-gemini.git
cd cmdrdata-gemini
pip install -e .[dev]
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cmdrdata_gemini

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
```

### Code Quality

```bash
# Format code
black cmdrdata_gemini/
isort cmdrdata_gemini/

# Type checking
mypy cmdrdata_gemini/

# Security scanning
safety check
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass (`pytest`)
6. Format your code (`black . && isort .`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://docs.cmdrdata.ai/gemini](https://docs.cmdrdata.ai/gemini)
- **Issues**: [GitHub Issues](https://github.com/cmdrdata-ai/cmdrdata-gemini/issues)
- **Support**: [support@cmdrdata.ai](mailto:support@cmdrdata.ai)

## 🔗 Related Projects

- **[cmdrdata-openai](https://github.com/cmdrdata-ai/cmdrdata-openai)** - Usage tracking for OpenAI
- **[cmdrdata-anthropic](https://github.com/cmdrdata-ai/cmdrdata-anthropic)** - Usage tracking for Anthropic Claude
- **[CMDR Data Platform](https://www.cmdrdata.ai)** - Complete LLM usage analytics

## 📈 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a complete list of changes and version history.

---

**Built with ❤️ by the CMDR Data team**

*Making AI usage tracking effortless and transparent.*