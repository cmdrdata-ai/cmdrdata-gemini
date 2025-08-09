"""
Tests for TrackedProxy and Gemini-specific tracking
"""

import time
from unittest.mock import Mock, patch

import pytest

from cmdrdata_gemini.proxy import (
    GEMINI_TRACK_METHODS,
    TrackedProxy,
    track_batch_embed_contents,
    track_batch_generate_content,
    track_classify_text,
    track_count_tokens,
    track_embed_content,
    track_generate_content,
    track_start_chat,
)


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
            kwargs={"model": "gemini-2.5-flash"},
        )

        # Verify tracking was called
        mock_tracker.track_usage_background.assert_called_once()
        call_args = mock_tracker.track_usage_background.call_args[1]
        assert call_args["customer_id"] == "customer-123"
        assert call_args["model"] == "gemini-2.5-flash"
        assert call_args["input_tokens"] == 15
        assert call_args["output_tokens"] == 25
        assert call_args["provider"] == "google"
        assert call_args["metadata"]["response_id"] == "resp_123"
        assert call_args["metadata"]["finish_reason"] == "STOP"
        assert call_args["metadata"]["safety_ratings"] is None

    def test_track_generate_content_model_prefix_removal(self, mock_gemini_response):
        """Test that 'models/' prefix is removed from model name"""
        mock_tracker = Mock()

        track_generate_content(
            result=mock_gemini_response,
            customer_id="customer-123",
            tracker=mock_tracker,
            method_name="models.generate_content",
            args=(),
            kwargs={"model": "models/gemini-2.5-flash"},
        )

        # Verify model name has prefix removed
        mock_tracker.track_usage_background.assert_called_once()
        call_args = mock_tracker.track_usage_background.call_args[1]
        assert call_args["model"] == "gemini-2.5-flash"

    def test_track_generate_content_no_customer_id(self, mock_gemini_response):
        """Test tracking without customer ID"""
        mock_tracker = Mock()

        with patch(
            "cmdrdata_gemini.proxy.get_effective_customer_id", return_value=None
        ):
            track_generate_content(
                result=mock_gemini_response,
                customer_id=None,
                tracker=mock_tracker,
                method_name="models.generate_content",
                args=(),
                kwargs={},
            )

        # Verify tracking was called with customer_id=None (new behavior allows tracking without customer_id)
        mock_tracker.track_usage_background.assert_called_once()
        call_args = mock_tracker.track_usage_background.call_args[1]
        assert call_args["customer_id"] is None

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
            kwargs={},
        )

        # Verify tracking was not called
        mock_tracker.track_usage_background.assert_not_called()

    def test_track_generate_content_extraction_failure(self):
        """Test graceful handling of data extraction failure"""
        mock_response = Mock()
        # Mock response that raises exception when accessing usage_metadata
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = Mock(
            side_effect=Exception("Access error")
        )
        mock_tracker = Mock()

        # Should not raise exception
        track_generate_content(
            result=mock_response,
            customer_id="customer-123",
            tracker=mock_tracker,
            method_name="models.generate_content",
            args=(),
            kwargs={},
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
            error_code="5",  # NOT_FOUND
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
            kwargs={"model": "gemini-2.5-flash"},
        )

        # Verify tracking was called
        mock_tracker.track_usage_background.assert_called_once()
        call_args = mock_tracker.track_usage_background.call_args[1]
        assert call_args["customer_id"] == "customer-123"
        assert call_args["model"] == "gemini-2.5-flash"
        assert call_args["input_tokens"] == 15
        assert call_args["output_tokens"] == 0  # No generation for count_tokens
        assert call_args["provider"] == "google"
        assert call_args["metadata"]["operation"] == "count_tokens"
        assert call_args["metadata"]["total_tokens"] == 15

    def test_track_count_tokens_no_customer_id(self, mock_count_tokens_response):
        """Test count_tokens tracking without customer ID"""
        mock_tracker = Mock()

        with patch(
            "cmdrdata_gemini.proxy.get_effective_customer_id", return_value=None
        ):
            track_count_tokens(
                result=mock_count_tokens_response,
                customer_id=None,
                tracker=mock_tracker,
                method_name="models.count_tokens",
                args=(),
                kwargs={},
            )

        # Verify tracking was called with customer_id=None (new behavior allows tracking without customer_id)
        mock_tracker.track_usage_background.assert_called_once()
        call_args = mock_tracker.track_usage_background.call_args[1]
        assert call_args["customer_id"] is None

    def test_track_embed_content_success(self):
        """Test successful tracking of embed_content"""
        # Set up embedding response mock
        mock_embed_result = Mock()
        mock_embed_result.embedding = Mock()
        mock_embed_result.embedding.values = [
            0.1,
            0.2,
            0.3,
            0.4,
        ]  # 4-dimensional embedding

        mock_tracker = Mock()

        with patch(
            "cmdrdata_gemini.proxy.get_effective_customer_id"
        ) as mock_get_customer:
            mock_get_customer.return_value = "test-customer"

            track_embed_content(
                result=mock_embed_result,
                customer_id="test-customer",
                tracker=mock_tracker,
                method_name="models.embed_content",
                args=(),
                kwargs={
                    "model": "text-embedding-004",
                    "content": "Test content for embedding",
                },
            )

            # Verify tracker was called
            mock_tracker.track_usage_background.assert_called_once()
            call_kwargs = mock_tracker.track_usage_background.call_args[1]

            assert call_kwargs["customer_id"] == "test-customer"
            assert call_kwargs["model"] == "text-embedding-004"
            assert call_kwargs["input_tokens"] == 4  # Estimated from content
            assert call_kwargs["output_tokens"] == 0
            assert call_kwargs["provider"] == "google"
            assert call_kwargs["metadata"]["operation"] == "embed_content"
            assert call_kwargs["metadata"]["embedding_dimensions"] == 4
            assert (
                call_kwargs["metadata"]["content_length"] == 26
            )  # len("Test content for embedding")

    def test_track_batch_embed_contents_success(self):
        """Test successful tracking of batch_embed_contents"""
        # Set up batch embedding response mock
        mock_batch_embed_result = Mock()
        mock_batch_embed_result.embeddings = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6]),
            Mock(embedding=[0.7, 0.8, 0.9]),
        ]

        mock_tracker = Mock()

        with patch(
            "cmdrdata_gemini.proxy.get_effective_customer_id"
        ) as mock_get_customer:
            mock_get_customer.return_value = "test-customer"

            track_batch_embed_contents(
                result=mock_batch_embed_result,
                customer_id="test-customer",
                tracker=mock_tracker,
                method_name="models.batch_embed_contents",
                args=(),
                kwargs={
                    "model": "text-embedding-004",
                    "requests": [
                        {"content": "Text 1"},
                        {"content": "Text 2"},
                        {"content": "Text 3"},
                    ],
                },
            )

            # Verify tracker was called
            mock_tracker.track_usage_background.assert_called_once()
            call_kwargs = mock_tracker.track_usage_background.call_args[1]

            assert call_kwargs["customer_id"] == "test-customer"
            assert call_kwargs["model"] == "text-embedding-004"
            assert call_kwargs["input_tokens"] == 4  # (6+6+6) // 4 = 18 // 4 = 4
            assert call_kwargs["output_tokens"] == 0
            assert call_kwargs["provider"] == "google"
            assert call_kwargs["metadata"]["operation"] == "batch_embed_contents"
            assert call_kwargs["metadata"]["batch_size"] == 3
            assert call_kwargs["metadata"]["embeddings_generated"] == 3

    def test_track_classify_text_success(self):
        """Test successful tracking of classify_text"""
        # Set up classification response mock
        mock_classify_result = Mock()
        mock_classify_result.categories = [Mock(name="positive", confidence=0.9)]
        mock_classify_result.confidence = 0.9

        mock_tracker = Mock()

        with patch(
            "cmdrdata_gemini.proxy.get_effective_customer_id"
        ) as mock_get_customer:
            mock_get_customer.return_value = "test-customer"

            track_classify_text(
                result=mock_classify_result,
                customer_id="test-customer",
                tracker=mock_tracker,
                method_name="models.classify_text",
                args=(),
                kwargs={
                    "model": "text-classification-004",
                    "text": "This is a great product!",
                },
            )

            # Verify tracker was called
            mock_tracker.track_usage_background.assert_called_once()
            call_kwargs = mock_tracker.track_usage_background.call_args[1]

            assert call_kwargs["customer_id"] == "test-customer"
            assert call_kwargs["model"] == "text-classification-004"
            assert call_kwargs["input_tokens"] == 6  # int(5 words * 1.3)
            assert call_kwargs["output_tokens"] == 0
            assert call_kwargs["provider"] == "google"
            assert call_kwargs["metadata"]["operation"] == "classify_text"
            assert (
                call_kwargs["metadata"]["text_length"] == 24
            )  # len("This is a great product!")
            assert call_kwargs["metadata"]["categories_count"] == 1
            assert call_kwargs["metadata"]["confidence"] == 0.9

    def test_track_batch_generate_content_success(self):
        """Test successful tracking of batch_generate_content"""
        # Set up batch generation response mock
        mock_batch_result = Mock()
        mock_batch_result.responses = [
            Mock(usage_metadata=Mock(prompt_token_count=10, candidates_token_count=15)),
            Mock(usage_metadata=Mock(prompt_token_count=8, candidates_token_count=12)),
        ]

        mock_tracker = Mock()

        with patch(
            "cmdrdata_gemini.proxy.get_effective_customer_id"
        ) as mock_get_customer:
            mock_get_customer.return_value = "test-customer"

            track_batch_generate_content(
                result=mock_batch_result,
                customer_id="test-customer",
                tracker=mock_tracker,
                method_name="models.batch_generate_content",
                args=(),
                kwargs={
                    "model": "gemini-2.5-flash",
                    "requests": [{"contents": "Prompt 1"}, {"contents": "Prompt 2"}],
                },
            )

            # Verify tracker was called
            mock_tracker.track_usage_background.assert_called_once()
            call_kwargs = mock_tracker.track_usage_background.call_args[1]

            assert call_kwargs["customer_id"] == "test-customer"
            assert call_kwargs["model"] == "gemini-2.5-flash"
            assert call_kwargs["input_tokens"] == 18  # Sum of batch inputs
            assert call_kwargs["output_tokens"] == 27  # Sum of batch outputs
            assert call_kwargs["provider"] == "google"
            assert call_kwargs["metadata"]["operation"] == "batch_generate_content"
            assert call_kwargs["metadata"]["batch_size"] == 2
            assert call_kwargs["metadata"]["total_requests"] == 2

    def test_track_start_chat_success(self):
        """Test successful tracking of start_chat"""
        # Set up chat session mock
        mock_chat_result = Mock()
        mock_chat_result.model = "gemini-2.5-flash"
        mock_chat_result.history = []

        mock_tracker = Mock()

        with patch(
            "cmdrdata_gemini.proxy.get_effective_customer_id"
        ) as mock_get_customer:
            mock_get_customer.return_value = "test-customer"

            track_start_chat(
                result=mock_chat_result,
                customer_id="test-customer",
                tracker=mock_tracker,
                method_name="models.start_chat",
                args=(),
                kwargs={"model": "gemini-2.5-flash", "history": []},
            )

            # Verify tracker was called
            mock_tracker.track_usage_background.assert_called_once()
            call_kwargs = mock_tracker.track_usage_background.call_args[1]

            assert call_kwargs["customer_id"] == "test-customer"
            assert call_kwargs["model"] == "gemini-2.5-flash"
            assert call_kwargs["input_tokens"] == 0
            assert call_kwargs["output_tokens"] == 0
            assert call_kwargs["provider"] == "google"
            assert call_kwargs["metadata"]["operation"] == "start_chat"
            assert call_kwargs["metadata"]["initial_history_length"] == 0
            assert call_kwargs["metadata"]["chat_model"] == "gemini-2.5-flash"

    def test_gemini_track_methods_configuration(self):
        """Test that GEMINI_TRACK_METHODS is configured correctly"""
        expected_methods = {
            "models.generate_content": track_generate_content,
            "models.batch_generate_content": track_batch_generate_content,
            "models.embed_content": track_embed_content,
            "models.batch_embed_contents": track_batch_embed_contents,
            "models.classify_text": track_classify_text,
            "models.start_chat": track_start_chat,
            "models.count_tokens": track_count_tokens,
        }

        for method_name, expected_function in expected_methods.items():
            assert method_name in GEMINI_TRACK_METHODS, f"Missing method: {method_name}"
            assert callable(
                GEMINI_TRACK_METHODS[method_name]
            ), f"Method {method_name} not callable"
            assert (
                GEMINI_TRACK_METHODS[method_name] == expected_function
            ), f"Wrong function for {method_name}"

        # Verify we have the expected total count
        assert (
            len(GEMINI_TRACK_METHODS) == 7
        ), f"Expected 7 methods, got {len(GEMINI_TRACK_METHODS)}"

    def test_proxy_integration_all_methods(self):
        """Test that all tracking methods work through proxy integration"""
        mock_client = Mock()
        mock_tracker = Mock()

        # Set up nested mock structure for Gemini client
        mock_client.models = Mock()
        mock_client.models.generate_content = Mock(return_value="generate_result")
        mock_client.models.batch_generate_content = Mock(
            return_value="batch_generate_result"
        )
        mock_client.models.embed_content = Mock(return_value="embed_result")
        mock_client.models.batch_embed_contents = Mock(
            return_value="batch_embed_result"
        )
        mock_client.models.classify_text = Mock(return_value="classify_result")
        mock_client.models.start_chat = Mock(return_value="chat_result")
        mock_client.models.count_tokens = Mock(return_value="count_tokens_result")

        proxy = TrackedProxy(
            client=mock_client, tracker=mock_tracker, track_methods=GEMINI_TRACK_METHODS
        )

        # Test each method through proxy
        test_cases = [
            (lambda: proxy.models.generate_content(), "generate_result"),
            (lambda: proxy.models.batch_generate_content(), "batch_generate_result"),
            (lambda: proxy.models.embed_content(), "embed_result"),
            (lambda: proxy.models.batch_embed_contents(), "batch_embed_result"),
            (lambda: proxy.models.classify_text(), "classify_result"),
            (lambda: proxy.models.start_chat(), "chat_result"),
            (lambda: proxy.models.count_tokens(), "count_tokens_result"),
        ]

        for test_func, expected_result in test_cases:
            result = test_func()
            assert result == expected_result


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
