# cmdrdata-gemini

[![CI](https://github.com/cmdrdata-ai/cmdrdata-gemini/workflows/CI/badge.svg)](https://github.com/cmdrdata-ai/cmdrdata-gemini/actions)
[![codecov](https://codecov.io/gh/cmdrdata-ai/cmdrdata-gemini/branch/main/graph/badge.svg)](https://codecov.io/gh/cmdrdata-ai/cmdrdata-gemini)
[![PyPI version](https://badge.fury.io/py/cmdrdata-gemini.svg)](https://badge.fury.io/py/cmdrdata-gemini)
[![Python Support](https://img.shields.io/pypi/pyversions/cmdrdata-gemini.svg)](https://pypi.org/project/cmdrdata-gemini/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The standard for AI customer intelligence - track every Gemini call by customer, feature, or any dimension**

Join hundreds of companies making customer-level AI tracking the default. One line of code to add complete visibility into your AI operations. Free during beta.

## 📊 Complete AI Intelligence Layer

`cmdrdata-gemini` is the missing analytics layer for your AI-powered application:

### **Track Everything That Matters**
- **Customer Intelligence** - Know exactly which customers use what features
- **Metadata Everything** - Tag usage by feature, experiment, team, region, or any dimension
- **Usage Patterns** - Understand how your AI is actually being used
- **Real-time Analytics** - Instant visibility into your AI operations

### **Built for Modern AI Apps**
- **One-line integration** - Drop-in replacement for Google GenAI SDK
- **Zero latency overhead** - Async tracking never blocks your API calls
- **Unlimited custom fields** - Track any metadata that matters to your business
- **Privacy first** - Your data never touches our servers (optional self-hosting)

### **What You Can Track**
- **Token usage** by customer, feature, experiment, or any dimension
- **Model usage** patterns (Gemini 1.5 Flash, Gemini 1.5 Pro, etc.)
- **Customer behavior** - Who uses what, when, and how much
- **Custom metadata** - Unlimited fields for your specific needs
- **Performance metrics** - Latency, errors, success rates by segment

### 💎 Advanced Analytics with Custom Metadata

Track arbitrary metadata with each API call to enable sophisticated analytics:

```python
# Example: AI-powered content generation with feature tracking
response = client.models.generate_content(
    model="gemini-1.5-pro",
    contents="Write a comprehensive guide about renewable energy...",
    customer_id="customer-123",
    # Custom metadata for analytics
    custom_metadata={
        "feature": "content_generation",
        "experiment_group": "gemini_pro_test",
        "content_type": "technical_guide",
        "user_segment": "enterprise",
        "session_id": "sess_xyz789"
    }
)

# Example: AI tutoring platform with learning analytics
response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents=complex_physics_problem,
    customer_id="customer-456",
    custom_metadata={
        "use_case": "educational_tutoring",
        "subject": "physics",
        "interaction_count": 5,
        "learning_path": "advanced_physics",
        "engagement_score": "high"
    }
)

# Example: Multi-modal analysis with usage patterns
response = client.models.generate_content(
    model="gemini-1.5-pro-vision",
    contents=[image_data, "Analyze this image"],
    customer_id="customer-789",
    custom_metadata={
        "modality": "vision_text",
        "workflow": "image_analysis",
        "api_version": "v2",
        "client_platform": "web",
        "feature_flag": "vision_enabled"
    }
)
```

**Intelligence Use Cases:**
- **Feature adoption**: Track which AI features customers actually use
- **A/B testing**: Compare model performance across experiment groups
- **Learning analytics**: Understand educational engagement patterns
- **Multi-modal insights**: Analyze usage across different modalities
- **Platform optimization**: Identify performance bottlenecks by platform
- **Product development**: Data-driven feature prioritization

## 🛡️ Production Ready

**Extremely robust and reliable** - Built for production environments with:

- **Resilient Tracking:** Gemini calls succeed even if tracking fails.
- **Non-blocking I/O:** Fire-and-forget tracking never slows down your application.
- **Automatic Retries:** Failed tracking attempts are automatically retried with exponential backoff.
- **Thread-Safe Context:** Safely track usage across multi-threaded and async applications.
- **Enterprise Security:** API key sanitization and input validation.

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
    model="gemini-1.5-flash",
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
        model="gemini-1.5-flash",
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
        model="gemini-1.5-flash",
        contents="Help me code"
    )
    # Automatically tracked for customer-123!

# Or pass customer_id directly
response = client.models.generate_content(
    model="gemini-1.5-flash",
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
export CMDRDATA_ENDPOINT="https://api.cmdrdata.ai/api/events"  # Optional
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
    cmdrdata_endpoint="https://your-custom-endpoint.com/api/events",
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
    "model": "gemini-1.5-flash",
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
    model="gemini-1.5-flash",
    contents="How many tokens is this?"
)
print(f"Token count: {token_count.total_tokens}")
```

### Disable Tracking for Specific Calls

```python
# Disable tracking for sensitive operations
response = client.models.generate_content(
    model="gemini-1.5-flash",
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
- **Support**: [spot@cmdrdata.ai](mailto:spot@cmdrdata.ai)

## 🔗 Related Projects

- **[cmdrdata-openai](https://github.com/cmdrdata-ai/cmdrdata-openai)** - Usage tracking for OpenAI
- **[cmdrdata-anthropic](https://github.com/cmdrdata-ai/cmdrdata-anthropic)** - Usage tracking for Anthropic Claude
- **[CMDR Data Platform](https://www.cmdrdata.ai)** - Complete LLM usage analytics

## 📈 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a complete list of changes and version history.

---

**Built with ❤️ by the CMDR Data team**

*Become the Google Analytics of your AI - understand everything, optimize everything.*
