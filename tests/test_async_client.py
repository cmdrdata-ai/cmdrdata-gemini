"""
Tests for AsyncTrackedGemini client
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from cmdrdata_gemini import AsyncTrackedGemini
from cmdrdata_gemini.exceptions import ConfigurationError, ValidationError


class TestAsyncTrackedGemini:
    def test_async_client_initialization_success(self):
        """Test successful async client initialization"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()

            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            )

            assert client is not None
            assert client._track_usage is True
            assert client._tracker is not None

    def test_async_client_initialization_without_tracking(self):
        """Test async client initialization without usage tracking"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()

            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            )

            assert client is not None
            assert client._track_usage is False
            assert client._tracker is None

    def test_missing_genai_sdk(self):
        """Test error when Google Gen AI SDK is not installed"""
        with patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(ConfigurationError, match="Google Gen AI SDK not found"):
                AsyncTrackedGemini()

    def test_invalid_genai_api_key(self):
        """Test validation of invalid Google Gen AI API key"""
        with patch("google.genai.Client"):
            with pytest.raises(ValidationError, match="Invalid Google Gen AI API key"):
                AsyncTrackedGemini(api_key="invalid-key")

    def test_invalid_cmdrdata_api_key(self):
        """Test validation of invalid cmdrdata API key"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()

            with pytest.raises(ValidationError, match="Invalid cmdrdata API key"):
                AsyncTrackedGemini(
                    api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    cmdrdata_api_key="invalid-key",
                )

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager functionality"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_client.__aenter__ = AsyncMock()
            mock_client.__aexit__ = AsyncMock()
            mock_genai.return_value = mock_client

            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            )

            async with client:
                pass

            mock_client.__aenter__.assert_called_once()
            mock_client.__aexit__.assert_called_once()

    def test_repr(self):
        """Test string representation of async client"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()

            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            )

            repr_str = repr(client)
            assert "AsyncTrackedGemini" in repr_str
            assert "enabled" in repr_str

    def test_version_compatibility_warning(self):
        """Test version compatibility warning"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()

            with patch(
                "cmdrdata_gemini.async_client.check_compatibility", return_value=False
            ):
                with patch("cmdrdata_gemini.async_client.logger") as mock_logger:
                    AsyncTrackedGemini(
                        api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                    )
                    mock_logger.warning.assert_called_once()

    def test_client_initialization_failure(self):
        """Test failure during client initialization"""
        with patch("google.genai.Client", side_effect=Exception("Init failed")):
            with pytest.raises(
                ConfigurationError, match="Failed to initialize Google Gen AI client"
            ):
                AsyncTrackedGemini(api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    def test_tracker_initialization_failure(self):
        """Test failure during tracker initialization"""
        with patch("google.genai.Client") as mock_genai:
            mock_genai.return_value = Mock()

            with patch(
                "cmdrdata_gemini.async_client.UsageTracker",
                side_effect=Exception("Tracker failed"),
            ):
                with patch("cmdrdata_gemini.async_client.logger") as mock_logger:
                    client = AsyncTrackedGemini(
                        api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                        cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
                    )
                    assert client._track_usage is False
                    mock_logger.warning.assert_called_once()

    def test_getattr_attribute_error(self):
        """Test __getattr__ method with non-existent attribute"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            del mock_client.nonexistent_attr  # Ensure it doesn't exist
            mock_genai.return_value = mock_client

            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            )

            with pytest.raises(
                AttributeError, match="object has no attribute 'nonexistent_attr'"
            ):
                getattr(client, "nonexistent_attr")

    def test_setattr_original_client(self):
        """Test __setattr__ method for setting attributes on original client"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_genai.return_value = mock_client

            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            )

            # Set a non-private attribute - should be forwarded to original client
            client.custom_attr = "test_value"
            assert mock_client.custom_attr == "test_value"

    def test_dir_method(self):
        """Test __dir__ method returns combined attributes"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_client.__dir__ = Mock(return_value=["models", "chat", "completions"])
            mock_genai.return_value = mock_client

            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            )

            attrs = dir(client)
            # Should include both proxy and original client attributes
            assert isinstance(attrs, list)
            # Check some expected attributes are present
            assert any("models" in str(attr) for attr in attrs)


class TestAsyncTrackedGeminiGenerateContent:
    @pytest.mark.asyncio
    async def test_async_generate_content_with_tracking_success(
        self, mock_gemini_response
    ):
        """Test successful async generate_content call with tracking"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.generate_content = Mock(return_value=mock_gemini_response)
            mock_client.models = mock_models
            mock_genai.return_value = mock_client

            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            )

            with patch.object(
                client._tracker, "track_usage_async", new_callable=AsyncMock
            ) as mock_track:
                result = await client.models.generate_content(
                    model="gemini-2.5-flash", contents="Hello, Gemini!"
                )

                # Verify original API was called
                mock_models.generate_content.assert_called_once()

                # Verify tracking was called
                mock_track.assert_called_once()
                call_args = mock_track.call_args[1]
                assert call_args["model"] == "gemini-2.5-flash"
                assert call_args["input_tokens"] == 15
                assert call_args["output_tokens"] == 25
                assert call_args["provider"] == "google"

                assert result == mock_gemini_response

    @pytest.mark.asyncio
    async def test_async_generate_content_without_tracking(self, mock_gemini_response):
        """Test async generate_content call without tracking enabled"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.generate_content = Mock(return_value=mock_gemini_response)
            mock_client.models = mock_models
            mock_genai.return_value = mock_client

            # Client without tracking
            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            )

            result = await client.models.generate_content(
                model="gemini-2.5-flash", contents="Hello, Gemini!"
            )

            # Verify original API was called
            mock_models.generate_content.assert_called_once()
            assert result == mock_gemini_response

    @pytest.mark.asyncio
    async def test_async_generate_content_with_customer_context(
        self, mock_gemini_response
    ):
        """Test async generate_content with customer context"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.generate_content = Mock(return_value=mock_gemini_response)
            mock_client.models = mock_models
            mock_genai.return_value = mock_client

            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            )

            with patch.object(
                client._tracker, "track_usage_async", new_callable=AsyncMock
            ) as mock_track:
                result = await client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Hello, Gemini!",
                    customer_id="customer-123",
                )

                # Verify tracking was called with customer ID
                mock_track.assert_called_once()
                call_args = mock_track.call_args[1]
                assert call_args["customer_id"] == "customer-123"

    @pytest.mark.asyncio
    async def test_async_generate_content_tracking_disabled(self, mock_gemini_response):
        """Test async generate_content with tracking explicitly disabled"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.generate_content = Mock(return_value=mock_gemini_response)
            mock_client.models = mock_models
            mock_genai.return_value = mock_client

            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            )

            with patch.object(
                client._tracker, "track_usage_async", new_callable=AsyncMock
            ) as mock_track:
                result = await client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Hello, Gemini!",
                    track_usage=False,
                )

                # Verify tracking was not called
                mock_track.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_generate_content_tracking_failure(self, mock_gemini_response):
        """Test that async API call succeeds even if tracking fails"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.generate_content = Mock(return_value=mock_gemini_response)
            mock_client.models = mock_models
            mock_genai.return_value = mock_client

            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            )

            with patch.object(
                client._tracker,
                "track_usage_async",
                new_callable=AsyncMock,
                side_effect=Exception("Tracking failed"),
            ):
                # Should not raise exception
                result = await client.models.generate_content(
                    model="gemini-2.5-flash", contents="Hello, Gemini!"
                )

                assert result == mock_gemini_response

    @pytest.mark.asyncio
    async def test_async_count_tokens_with_tracking_success(
        self, mock_count_tokens_response
    ):
        """Test successful async count_tokens call with tracking"""
        with patch("google.genai.Client") as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.count_tokens = Mock(return_value=mock_count_tokens_response)
            mock_client.models = mock_models
            mock_genai.return_value = mock_client

            client = AsyncTrackedGemini(
                api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                cmdrdata_api_key="tk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            )

            with patch.object(
                client._tracker, "track_usage_async", new_callable=AsyncMock
            ) as mock_track:
                result = await client.models.count_tokens(
                    model="gemini-2.5-flash", contents="Hello, Gemini!"
                )

                # Verify original API was called
                mock_models.count_tokens.assert_called_once()

                # Verify tracking was called
                mock_track.assert_called_once()
                call_args = mock_track.call_args[1]
                assert call_args["model"] == "gemini-2.5-flash"
                assert call_args["input_tokens"] == 15
                assert call_args["output_tokens"] == 0  # No generation for count_tokens
                assert call_args["provider"] == "google"

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
