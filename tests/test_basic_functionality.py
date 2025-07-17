"""
Basic integration tests for cmdrdata-gemini
"""

from unittest.mock import Mock, patch

import pytest

from cmdrdata_gemini import TrackedGemini, AsyncTrackedGemini
from cmdrdata_gemini.context import customer_context


class TestBasicFunctionality:
    def test_import_tracked_gemini(self):
        """Test that TrackedGemini can be imported"""
        assert TrackedGemini is not None

    def test_import_async_tracked_gemini(self):
        """Test that AsyncTrackedGemini can be imported"""
        assert AsyncTrackedGemini is not None

    def test_basic_client_creation(self):
        """Test basic client creation without errors"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()
            
            client = TrackedGemini(api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            assert client is not None

    def test_basic_async_client_creation(self):
        """Test basic async client creation without errors"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()
            
            client = AsyncTrackedGemini(api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            assert client is not None

    def test_client_with_tracking_enabled(self):
        """Test client creation with tracking enabled"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()
            
            client = TrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            )
            
            assert client._track_usage is True
            assert client._tracker is not None

    def test_customer_context_integration(self, mock_gemini_response):
        """Test integration with customer context"""
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
                with customer_context("customer-123"):
                    result = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents="Hello, Gemini!"
                    )
                
                # Verify tracking was called with context customer ID
                mock_track.assert_called_once()
                call_args = mock_track.call_args[1]
                assert call_args["customer_id"] == "customer-123"

    def test_version_info_available(self):
        """Test that version information is available"""
        from cmdrdata_gemini import get_version
        
        version = get_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_compatibility_check_available(self):
        """Test that compatibility checking is available"""
        from cmdrdata_gemini import check_compatibility, get_compatibility_info
        
        compat = check_compatibility()
        assert isinstance(compat, bool)
        
        info = get_compatibility_info()
        assert isinstance(info, dict)
        assert "google_genai" in info
        assert "python" in info

    def test_exceptions_available(self):
        """Test that exceptions are properly exported"""
        from cmdrdata_gemini import (
            CMDRDataError,
            ValidationError,
            ConfigurationError,
            NetworkError,
            TrackingError,
        )
        
        # All should be classes
        assert isinstance(CMDRDataError, type)
        assert isinstance(ValidationError, type)
        assert isinstance(ConfigurationError, type)
        assert isinstance(NetworkError, type)
        assert isinstance(TrackingError, type)

    def test_context_functions_available(self):
        """Test that context management functions are available"""
        from cmdrdata_gemini import (
            customer_context,
            set_customer_context,
            get_customer_context,
            clear_customer_context,
        )
        
        # Test basic context operations
        set_customer_context("test-customer")
        assert get_customer_context() == "test-customer"
        
        clear_customer_context()
        assert get_customer_context() is None

    @pytest.mark.asyncio
    async def test_async_basic_integration(self, mock_gemini_response):
        """Test basic async integration"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.generate_content = Mock(return_value=mock_gemini_response)
            mock_client.models = mock_models
            mock_genai.return_value = mock_client
            
            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            )
            
            # Should be able to access models
            models = client.models
            assert models is not None


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