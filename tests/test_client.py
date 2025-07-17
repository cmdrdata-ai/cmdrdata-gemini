"""
Tests for TrackedGemini client
"""

from unittest.mock import Mock, patch

import pytest

from cmdrdata_gemini import TrackedGemini
from cmdrdata_gemini.exceptions import ConfigurationError, ValidationError


class TestTrackedGemini:
    def test_client_initialization_success(self):
        """Test successful client initialization"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()
            
            client = TrackedGemini(
                api_key="AIza" + "A" * 35,
                cmdrdata_api_key="tk-" + "C" * 32
            )
            
            assert client is not None
            assert client._track_usage is True
            assert client._tracker is not None

    def test_client_initialization_without_tracking(self):
        """Test client initialization without usage tracking"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()
            
            client = TrackedGemini(api_key="AIza" + "A" * 35)
            
            assert client is not None
            assert client._track_usage is False
            assert client._tracker is None

    def test_client_initialization_disabled_tracking(self):
        """Test client initialization with explicitly disabled tracking"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()
            
            client = TrackedGemini(
                api_key="AIza" + "A" * 35,
                cmdrdata_api_key="tk-" + "C" * 32,
                track_usage=False
            )
            
            assert client is not None
            assert client._track_usage is False

    def test_missing_genai_sdk(self):
        """Test error when Google Gen AI SDK is not installed"""
        with patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(ConfigurationError, match="Google Gen AI SDK not found"):
                TrackedGemini()

    def test_invalid_genai_api_key(self):
        """Test validation of invalid Google Gen AI API key"""
        with patch("google.genai.Client"):
            with pytest.raises(ValidationError, match="Invalid Google Gen AI API key"):
                TrackedGemini(api_key="invalid-key")

    def test_invalid_cmdrdata_api_key(self):
        """Test validation of invalid cmdrdata API key"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()
            
            with pytest.raises(ValidationError, match="Invalid cmdrdata API key"):
                TrackedGemini(
                    api_key="AIza" + "A" * 35,
                    cmdrdata_api_key="invalid-key"
                )

    def test_genai_client_failure(self):
        """Test handling of Google Gen AI client initialization failure"""
        with patch("google.genai.Client", side_effect=Exception("Client error")):
            with pytest.raises(ConfigurationError, match="Failed to initialize Google Gen AI client"):
                TrackedGemini(api_key="AIza" + "A" * 35)

    def test_tracker_initialization_failure(self):
        """Test graceful handling of tracker initialization failure"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()
            
            with patch("cmdrdata_gemini.client.UsageTracker", side_effect=Exception("Tracker error")):
                client = TrackedGemini(
                    api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
                )
                
                # Should still create client but disable tracking
                assert client is not None
                assert client._track_usage is False

    def test_attribute_forwarding(self):
        """Test that client attributes are forwarded correctly"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_client.some_attribute = "test_value"
            mock_genai.return_value = mock_client
            
            client = TrackedGemini(api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            
            assert client.some_attribute == "test_value"

    def test_tracked_models_access(self):
        """Test accessing models attribute returns tracked proxy"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_client.models = mock_models
            mock_genai.return_value = mock_client
            
            client = TrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            )
            
            # Should return a TrackedProxy for models
            models = client.models
            assert models is not None
            # The proxy should wrap the original models object

    def test_get_usage_tracker(self):
        """Test getting the usage tracker instance"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()
            
            client = TrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            )
            
            tracker = client.get_usage_tracker()
            assert tracker is not None

    def test_get_performance_stats(self):
        """Test getting performance statistics"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()
            
            client = TrackedGemini(api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            
            stats = client.get_performance_stats()
            assert isinstance(stats, dict)

    def test_repr(self):
        """Test string representation of client"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()
            
            client = TrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            )
            
            repr_str = repr(client)
            assert "TrackedGemini" in repr_str
            assert "enabled" in repr_str


class TestTrackedGeminiGenerateContent:
    def test_generate_content_with_tracking_success(self, mock_gemini_response):
        """Test successful generate_content call with tracking"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.generate_content.return_value = mock_gemini_response
            mock_client.models = mock_models
            mock_genai.return_value = mock_client
            
            client = TrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            )
            
            with patch.object(client._tracker, 'track_usage_background') as mock_track:
                result = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Hello, Gemini!"
                )
                
                # Verify original API was called
                mock_models.generate_content.assert_called_once()
                
                # Verify tracking was called
                mock_track.assert_called_once_with(
                    customer_id=None,  # No context set
                    model="gemini-2.5-flash",
                    input_tokens=15,
                    output_tokens=25,
                    provider="google",
                    metadata={
                        "response_id": "resp_123",
                        "model_version": "001",
                        "safety_ratings": None,
                        "finish_reason": "STOP",
                        "total_token_count": 40,
                    }
                )
                
                assert result == mock_gemini_response

    def test_generate_content_without_tracking(self, mock_gemini_response):
        """Test generate_content call without tracking enabled"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.generate_content.return_value = mock_gemini_response
            mock_client.models = mock_models
            mock_genai.return_value = mock_client
            
            # Client without tracking
            client = TrackedGemini(api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            
            result = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Hello, Gemini!"
            )
            
            # Verify original API was called
            mock_models.generate_content.assert_called_once()
            assert result == mock_gemini_response

    def test_generate_content_with_customer_context(self, mock_gemini_response):
        """Test generate_content with customer context"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.generate_content.return_value = mock_gemini_response
            mock_client.models = mock_models
            mock_genai.return_value = mock_client
            
            client = TrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            )
            
            with patch.object(client._tracker, 'track_usage_background') as mock_track:
                result = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Hello, Gemini!",
                    customer_id="customer-123"
                )
                
                # Verify tracking was called with customer ID
                mock_track.assert_called_once()
                call_args = mock_track.call_args[1]
                assert call_args["customer_id"] == "customer-123"

    def test_generate_content_tracking_disabled(self, mock_gemini_response):
        """Test generate_content with tracking explicitly disabled"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.generate_content.return_value = mock_gemini_response
            mock_client.models = mock_models
            mock_genai.return_value = mock_client
            
            client = TrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            )
            
            with patch.object(client._tracker, 'track_usage_background') as mock_track:
                result = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Hello, Gemini!",
                    track_usage=False
                )
                
                # Verify tracking was not called
                mock_track.assert_not_called()

    def test_generate_content_tracking_failure(self, mock_gemini_response):
        """Test that API call succeeds even if tracking fails"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.generate_content.return_value = mock_gemini_response
            mock_client.models = mock_models
            mock_genai.return_value = mock_client
            
            client = TrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            )
            
            with patch.object(client._tracker, 'track_usage_background', side_effect=Exception("Tracking failed")):
                # Should not raise exception
                result = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Hello, Gemini!"
                )
                
                assert result == mock_gemini_response

    def test_count_tokens_with_tracking_success(self, mock_count_tokens_response):
        """Test successful count_tokens call with tracking"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.count_tokens.return_value = mock_count_tokens_response
            mock_client.models = mock_models
            mock_genai.return_value = mock_client
            
            client = TrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            )
            
            with patch.object(client._tracker, 'track_usage_background') as mock_track:
                result = client.models.count_tokens(
                    model="gemini-2.5-flash",
                    contents="Hello, Gemini!"
                )
                
                # Verify original API was called
                mock_models.count_tokens.assert_called_once()
                
                # Verify tracking was called
                mock_track.assert_called_once_with(
                    customer_id=None,  # No context set
                    model="gemini-2.5-flash",
                    input_tokens=15,
                    output_tokens=0,  # No generation for count_tokens
                    provider="google",
                    metadata={
                        "operation": "count_tokens",
                        "total_tokens": 15,
                    }
                )
                
                assert result == mock_count_tokens_response


@pytest.fixture
def mock_gemini_response():
    """Mock Google Gen AI generate_content response"""
    response = Mock()
    response.id = "resp_123"
    response.model_version = "001"
    response.safety_ratings = None
    response.text = "Hello! How can I help you today?"
    
    # Mock candidates
    candidate = Mock()
    candidate.finish_reason = "STOP"
    response.candidates = [candidate]
    
    # Mock usage metadata
    response.usage_metadata = Mock()
    response.usage_metadata.prompt_token_count = 15
    response.usage_metadata.candidates_token_count = 25
    response.usage_metadata.total_token_count = 40
    
    return response


@pytest.fixture
def mock_count_tokens_response():
    """Mock Google Gen AI count_tokens response"""
    response = Mock()
    response.total_tokens = 15
    
    return response