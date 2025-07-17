"""
Tests for TrackedProxy and Gemini-specific tracking
"""

import time
from unittest.mock import Mock, patch

import pytest

from cmdrdata_gemini.proxy import GEMINI_TRACK_METHODS, TrackedProxy, track_generate_content, track_count_tokens


class TestTrackedProxy:
    def test_proxy_forwards_attributes(self):
        """Test that proxy forwards attribute access to underlying client"""
        mock_client = Mock()
        mock_client.some_attr = "test_value"
        mock_tracker = Mock()
        
        proxy = TrackedProxy(mock_client, mock_tracker, {})
        
        assert proxy.some_attr == "test_value"

    def test_proxy_forwards_method_calls(self):
        """Test that proxy forwards method calls to underlying client"""
        mock_client = Mock()
        mock_client.some_method.return_value = "result"
        mock_tracker = Mock()
        
        proxy = TrackedProxy(mock_client, mock_tracker, {})
        
        result = proxy.some_method("arg1", kwarg="value")
        assert result == "result"
        mock_client.some_method.assert_called_once_with("arg1", kwarg="value")

    def test_proxy_wraps_tracked_methods(self):
        """Test that proxy wraps methods that should be tracked"""
        mock_client = Mock()
        mock_client.tracked_method.return_value = "result"
        mock_tracker = Mock()
        mock_track_func = Mock()
        
        track_methods = {"tracked_method": mock_track_func}
        proxy = TrackedProxy(mock_client, mock_tracker, track_methods)
        
        result = proxy.tracked_method("arg1", kwarg="value")
        
        # Verify original method was called
        mock_client.tracked_method.assert_called_once_with("arg1", kwarg="value")
        
        # Verify tracking function was called
        mock_track_func.assert_called_once()
        assert result == "result"

    def test_proxy_handles_nested_attributes(self):
        """Test that proxy handles nested attributes like client.models.generate_content"""
        mock_client = Mock()
        mock_models = Mock()
        mock_models.generate_content.return_value = "result"
        mock_client.models = mock_models
        mock_tracker = Mock()
        mock_track_func = Mock()
        
        track_methods = {"models.generate_content": mock_track_func}
        proxy = TrackedProxy(mock_client, mock_tracker, track_methods)
        
        # Access nested attribute
        models_proxy = proxy.models
        assert models_proxy is not None
        
        # Call the nested method
        result = models_proxy.generate_content("arg1", kwarg="value")
        
        # Verify original method was called
        mock_models.generate_content.assert_called_once_with("arg1", kwarg="value")
        
        # Verify tracking function was called
        mock_track_func.assert_called_once()
        assert result == "result"

    def test_proxy_customer_id_extraction(self):
        """Test that proxy extracts customer_id from kwargs"""
        mock_client = Mock()
        mock_client.tracked_method.return_value = "result"
        mock_tracker = Mock()
        mock_track_func = Mock()
        
        track_methods = {"tracked_method": mock_track_func}
        proxy = TrackedProxy(mock_client, mock_tracker, track_methods)
        
        result = proxy.tracked_method("arg1", customer_id="customer-123", kwarg="value")
        
        # Verify customer_id was removed from kwargs before calling original method
        mock_client.tracked_method.assert_called_once_with("arg1", kwarg="value")
        
        # Verify tracking function received customer_id
        mock_track_func.assert_called_once()
        call_kwargs = mock_track_func.call_args[1]
        assert call_kwargs["customer_id"] == "customer-123"

    def test_proxy_tracking_disabled(self):
        """Test that proxy respects track_usage=False"""
        mock_client = Mock()
        mock_client.tracked_method.return_value = "result"
        mock_tracker = Mock()
        mock_track_func = Mock()
        
        track_methods = {"tracked_method": mock_track_func}
        proxy = TrackedProxy(mock_client, mock_tracker, track_methods)
        
        result = proxy.tracked_method("arg1", track_usage=False, kwarg="value")
        
        # Verify original method was called
        mock_client.tracked_method.assert_called_once_with("arg1", kwarg="value")
        
        # Verify tracking function was NOT called
        mock_track_func.assert_not_called()

    def test_proxy_tracking_failure_resilience(self):
        """Test that proxy continues if tracking fails"""
        mock_client = Mock()
        mock_client.tracked_method.return_value = "result"
        mock_tracker = Mock()
        mock_track_func = Mock(side_effect=Exception("Tracking failed"))
        
        track_methods = {"tracked_method": mock_track_func}
        proxy = TrackedProxy(mock_client, mock_tracker, track_methods)
        
        # Should not raise exception
        result = proxy.tracked_method("arg1", kwarg="value")
        
        # Verify original method was called and result returned
        mock_client.tracked_method.assert_called_once_with("arg1", kwarg="value")
        assert result == "result"

    def test_proxy_tracks_api_error(self):
        """Test that the proxy tracks an error if the API call fails"""
        mock_client = Mock()
        # Simulate an API error from the client
        api_error = Exception("API call failed")
        mock_client.tracked_method.side_effect = api_error
        
        mock_tracker = Mock()
        mock_track_func = Mock()
        
        track_methods = {"tracked_method": mock_track_func}
        proxy = TrackedProxy(mock_client, mock_tracker, track_methods)
        
        # The proxy should re-raise the original exception
        with pytest.raises(Exception, match="API call failed"):
            proxy.tracked_method("arg1", kwarg="value")
        
        # Verify that the tracking function was still called with error details
        mock_track_func.assert_called_once()
        call_kwargs = mock_track_func.call_args[1]
        
        assert call_kwargs["result"] is None
        assert call_kwargs["error_occurred"] is True
        assert call_kwargs["error_type"] == "sdk_error"
        assert "API call failed" in call_kwargs["error_message"]
        assert call_kwargs["request_start_time"] is not None
        assert call_kwargs["request_end_time"] is not None

    def test_proxy_attribute_error(self):
        """Test that proxy raises AttributeError for non-existent attributes"""
        mock_client = Mock()
        del mock_client.nonexistent_attr  # Ensure it doesn't exist
        mock_tracker = Mock()
        
        proxy = TrackedProxy(mock_client, mock_tracker, {})
        
        with pytest.raises(AttributeError):
            _ = proxy.nonexistent_attr

    def test_proxy_dir(self):
        """Test that proxy __dir__ returns attributes from both proxy and client"""
        mock_client = Mock()
        mock_client.client_attr = "value"
        mock_tracker = Mock()
        
        proxy = TrackedProxy(mock_client, mock_tracker, {})
        
        dir_result = dir(proxy)
        assert "client_attr" in dir_result

    def test_proxy_repr(self):
        """Test proxy string representation"""
        mock_client = Mock()
        mock_tracker = Mock()
        
        proxy = TrackedProxy(mock_client, mock_tracker, {})
        
        repr_str = repr(proxy)
        assert "TrackedProxy" in repr_str


class TestGeminiTrackingMethods:
    def test_track_generate_content_success(self, mock_gemini_response):
        """Test successful tracking of generate_content"""
        mock_tracker = Mock()
        
        track_generate_content(
            result=mock_gemini_response,
            customer_id="customer-123",
            tracker=mock_tracker,
            method_name="models.generate_content",
            args=(),
            kwargs={"model": "gemini-2.5-flash"}
        )
        
        # Verify tracking was called
        mock_tracker.track_usage_background.assert_called_once_with(
            customer_id="customer-123",
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

    def test_track_generate_content_model_prefix_removal(self, mock_gemini_response):
        """Test that 'models/' prefix is removed from model name"""
        mock_tracker = Mock()
        
        track_generate_content(
            result=mock_gemini_response,
            customer_id="customer-123",
            tracker=mock_tracker,
            method_name="models.generate_content",
            args=(),
            kwargs={"model": "models/gemini-2.5-flash"}
        )
        
        # Verify model name has prefix removed
        mock_tracker.track_usage_background.assert_called_once()
        call_args = mock_tracker.track_usage_background.call_args[1]
        assert call_args["model"] == "gemini-2.5-flash"

    def test_track_generate_content_no_customer_id(self, mock_gemini_response):
        """Test tracking without customer ID"""
        mock_tracker = Mock()
        
        with patch("cmdrdata_gemini.proxy.get_effective_customer_id", return_value=None):
            track_generate_content(
                result=mock_gemini_response,
                customer_id=None,
                tracker=mock_tracker,
                method_name="models.generate_content",
                args=(),
                kwargs={}
            )
        
        # Verify tracking was not called
        mock_tracker.track_usage_background.assert_not_called()

    def test_track_generate_content_no_usage_info(self):
        """Test tracking with response that has no usage info"""
        mock_response = Mock()
        del mock_response.usage_metadata  # No usage_metadata attribute
        mock_tracker = Mock()
        
        track_generate_content(
            result=mock_response,
            customer_id="customer-123",
            tracker=mock_tracker,
            method_name="models.generate_content",
            args=(),
            kwargs={}
        )
        
        # Verify tracking was not called
        mock_tracker.track_usage_background.assert_not_called()

    def test_track_generate_content_extraction_failure(self):
        """Test graceful handling of data extraction failure"""
        mock_response = Mock()
        # Mock response that raises exception when accessing usage_metadata
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = Mock(side_effect=Exception("Access error"))
        mock_tracker = Mock()
        
        # Should not raise exception
        track_generate_content(
            result=mock_response,
            customer_id="customer-123",
            tracker=mock_tracker,
            method_name="models.generate_content",
            args=(),
            kwargs={}
        )
        
        # Verify tracking was not called due to error
        mock_tracker.track_usage_background.assert_not_called()

    def test_track_generate_content_with_error(self):
        """Test tracking of a failed generate_content call"""
        mock_tracker = Mock()
        start_time = time.time() - 1
        end_time = time.time()

        track_generate_content(
            result=None,
            customer_id="customer-123",
            tracker=mock_tracker,
            method_name="models.generate_content",
            args=(),
            kwargs={"model": "gemini-2.5-flash"},
            error_occurred=True,
            error_type="grpc_error",
            error_code="5", # NOT_FOUND
            error_message="Model not found",
            request_id="req_xyz",
            request_start_time=start_time,
            request_end_time=end_time,
        )

        mock_tracker.track_usage_background.assert_called_once()
        call_kwargs = mock_tracker.track_usage_background.call_args[1]

        assert call_kwargs["customer_id"] == "customer-123"
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["input_tokens"] == 0
        assert call_kwargs["output_tokens"] == 0
        assert call_kwargs["provider"] == "google"
        assert call_kwargs["error_occurred"] is True
        assert call_kwargs["error_type"] == "grpc_error"
        assert call_kwargs["error_code"] == "5"
        assert call_kwargs["error_message"] == "Model not found"
        assert call_kwargs["request_id"] == "req_xyz"
        assert call_kwargs["request_start_time"] == start_time
        assert call_kwargs["request_end_time"] == end_time

    def test_track_count_tokens_success(self, mock_count_tokens_response):
        """Test successful tracking of count_tokens"""
        mock_tracker = Mock()
        
        track_count_tokens(
            result=mock_count_tokens_response,
            customer_id="customer-123",
            tracker=mock_tracker,
            method_name="models.count_tokens",
            args=(),
            kwargs={"model": "gemini-2.5-flash"}
        )
        
        # Verify tracking was called
        mock_tracker.track_usage_background.assert_called_once_with(
            customer_id="customer-123",
            model="gemini-2.5-flash",
            input_tokens=15,
            output_tokens=0,  # No generation for count_tokens
            provider="google",
            metadata={
                "operation": "count_tokens",
                "total_tokens": 15,
            }
        )

    def test_track_count_tokens_no_customer_id(self, mock_count_tokens_response):
        """Test count_tokens tracking without customer ID"""
        mock_tracker = Mock()
        
        with patch("cmdrdata_gemini.proxy.get_effective_customer_id", return_value=None):
            track_count_tokens(
                result=mock_count_tokens_response,
                customer_id=None,
                tracker=mock_tracker,
                method_name="models.count_tokens",
                args=(),
                kwargs={}
            )
        
        # Verify tracking was not called
        mock_tracker.track_usage_background.assert_not_called()

    def test_gemini_track_methods_configuration(self):
        """Test that GEMINI_TRACK_METHODS is configured correctly"""
        assert "models.generate_content" in GEMINI_TRACK_METHODS
        assert "models.count_tokens" in GEMINI_TRACK_METHODS
        assert GEMINI_TRACK_METHODS["models.generate_content"] == track_generate_content
        assert GEMINI_TRACK_METHODS["models.count_tokens"] == track_count_tokens


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