"""
Dynamic proxy classes for transparent API forwarding with usage tracking
"""

import inspect
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union

from .context import get_effective_customer_id
from .tracker import UsageTracker

logger = logging.getLogger(__name__)


class TrackedProxy:
    """
    Base proxy class that forwards all method calls to the underlying client
    while selectively adding usage tracking to specific methods.
    """

    def __init__(
        self,
        client: Any,
        tracker: UsageTracker,
        track_methods: Dict[str, Callable] = None,
    ):
        """
        Initialize the proxy.

        Args:
            client: The underlying client (e.g., Google Gen AI client)
            tracker: Usage tracker instance
            track_methods: Dict mapping method names to tracking functions
        """
        # Store these with underscore prefixes to avoid conflicts
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_tracker", tracker)
        object.__setattr__(self, "_track_methods", track_methods or {})
        object.__setattr__(self, "_tracked_attributes", {})

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically forward attribute access to the underlying client.
        If the attribute is a method that should be tracked, wrap it.
        """
        # Check if we've already wrapped this attribute
        if name in self._tracked_attributes:
            return self._tracked_attributes[name]

        # Get the attribute from the underlying client
        try:
            attr = getattr(self._client, name)
        except AttributeError:
            # Don't cache non-existent attributes
            raise AttributeError(
                f"'{type(self._client).__name__}' object has no attribute '{name}'"
            )

        # If it's a callable and we have a tracker for it, wrap it
        if callable(attr) and name in self._track_methods:
            wrapped_attr = self._wrap_method(attr, name)
            self._tracked_attributes[name] = wrapped_attr
            return wrapped_attr

        # If it's another object that might need proxying, check if we should wrap it
        elif hasattr(attr, "__dict__") and not isinstance(
            attr, (str, int, float, bool, type(None))
        ):
            # This might be a sub-client (like client.models)
            # Check if any of our track methods start with this attribute name
            sub_track_methods = {
                k[len(name) + 1 :]: v
                for k, v in self._track_methods.items()
                if k.startswith(f"{name}.")
            }

            if sub_track_methods:
                wrapped_attr = TrackedProxy(attr, self._tracker, sub_track_methods)
                self._tracked_attributes[name] = wrapped_attr
                return wrapped_attr

        # For everything else, just return the original attribute
        self._tracked_attributes[name] = attr
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute setting to the underlying client"""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._client, name, value)

    def __dir__(self):
        """Return attributes from both proxy and underlying client"""
        proxy_attrs = [
            attr for attr in object.__dir__(self) if not attr.startswith("_")
        ]
        client_attrs = dir(self._client)
        return sorted(set(proxy_attrs + client_attrs))

    def _wrap_method(self, method: Callable, method_name: str) -> Callable:
        """Wrap a method to add usage tracking and performance monitoring"""
        tracker_func = self._track_methods[method_name]

        def wrapped(*args, **kwargs):
            customer_id = kwargs.pop("customer_id", ...)
            track_usage = kwargs.pop("track_usage", True)
            custom_metadata = kwargs.pop("metadata", None)

            request_id = str(uuid.uuid4())
            start_time = time.time()
            error_occurred = False
            error_type = None
            error_code = None
            error_message = None

            try:
                result = method(*args, **kwargs)
                end_time = time.time()

                if track_usage:
                    try:
                        tracker_func(
                            result=result,
                            customer_id=customer_id,
                            tracker=self._tracker,
                            method_name=method_name,
                            args=args,
                            kwargs=kwargs,
                            custom_metadata=custom_metadata,
                            request_start_time=start_time,
                            request_end_time=end_time,
                            error_occurred=error_occurred,
                            error_type=error_type,
                            error_code=error_code,
                            error_message=error_message,
                            request_id=request_id,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to track usage for {method_name}: {e}")

                return result

            except Exception as e:
                end_time = time.time()
                error_occurred = True
                error_message = str(e)

                # Google's SDK uses grpc, so we inspect the exception differently
                if hasattr(e, "code"):  # Heuristic for gRPC error
                    error_code = str(e.code())
                    error_type = "grpc_error"
                else:
                    error_type = "sdk_error"

                if track_usage:
                    try:
                        tracker_func(
                            result=None,
                            customer_id=customer_id,
                            tracker=self._tracker,
                            method_name=method_name,
                            args=args,
                            kwargs=kwargs,
                            custom_metadata=custom_metadata,
                            request_start_time=start_time,
                            request_end_time=end_time,
                            error_occurred=error_occurred,
                            error_type=error_type,
                            error_code=error_code,
                            error_message=error_message,
                            request_id=request_id,
                        )
                    except Exception as track_error:
                        logger.warning(
                            f"Failed to track error for {method_name}: {track_error}"
                        )

                raise

        wrapped.__name__ = getattr(method, "__name__", method_name)
        wrapped.__doc__ = getattr(method, "__doc__", None)
        try:
            wrapped.__signature__ = inspect.signature(method)
        except (ValueError, TypeError):
            pass

        return wrapped

    def __repr__(self):
        """Return a helpful representation"""
        return f"TrackedProxy({repr(self._client)})"


def track_generate_content(
    result,
    customer_id,
    tracker,
    method_name,
    args,
    kwargs,
    custom_metadata=None,
    # Enhanced tracking parameters
    request_start_time=None,
    request_end_time=None,
    error_occurred=None,
    error_type=None,
    error_code=None,
    error_message=None,
    request_id=None,
):
    """Track Google Gen AI generate_content usage"""
    try:
        effective_customer_id = get_effective_customer_id(customer_id)

        input_tokens = 0
        output_tokens = 0
        model = kwargs.get("model", "unknown")
        if isinstance(model, str) and model.startswith("models/"):
            model = model[7:]

        metadata = {}
        if result and hasattr(result, "usage_metadata"):
            input_tokens = getattr(result.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(result.usage_metadata, "candidates_token_count", 0)
            metadata.update(
                {
                    "response_id": getattr(result, "id", None),
                    "safety_ratings": getattr(result, "safety_ratings", None),
                    "finish_reason": (
                        getattr(result.candidates[0], "finish_reason", None)
                        if hasattr(result, "candidates") and result.candidates
                        else None
                    ),
                }
            )
        elif not error_occurred:
            # No usage data available and no error, skip tracking
            return

        if custom_metadata:
            metadata.update(custom_metadata)

        tracker.track_usage_background(
            customer_id=effective_customer_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider="google",
            metadata=metadata,
            request_start_time=request_start_time,
            request_end_time=request_end_time,
            error_occurred=error_occurred,
            error_type=error_type,
            error_code=error_code,
            error_message=error_message,
            request_id=request_id,
        )

    except Exception as e:
        logger.warning(f"Failed to extract usage data from generate_content: {e}")


def track_count_tokens(
    result,
    customer_id,
    tracker,
    method_name,
    args,
    kwargs,
    custom_metadata=None,
    # Enhanced tracking parameters
    request_start_time=None,
    request_end_time=None,
    error_occurred=None,
    error_type=None,
    error_code=None,
    error_message=None,
    request_id=None,
):
    """Track Google Gen AI count_tokens usage"""
    try:
        effective_customer_id = get_effective_customer_id(customer_id)

        input_tokens = 0
        model = kwargs.get("model", "unknown")
        if isinstance(model, str) and model.startswith("models/"):
            model = model[7:]

        metadata = {"operation": "count_tokens"}
        if result and hasattr(result, "total_tokens"):
            input_tokens = result.total_tokens
            metadata["total_tokens"] = result.total_tokens

        if custom_metadata:
            metadata.update(custom_metadata)

        tracker.track_usage_background(
            customer_id=effective_customer_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=0,
            provider="google",
            metadata=metadata,
            request_start_time=request_start_time,
            request_end_time=request_end_time,
            error_occurred=error_occurred,
            error_type=error_type,
            error_code=error_code,
            error_message=error_message,
            request_id=request_id,
        )

    except Exception as e:
        logger.warning(f"Failed to extract usage data from count_tokens: {e}")


def track_embed_content(
    result,
    customer_id,
    tracker,
    method_name,
    args,
    kwargs,
    custom_metadata=None,
    **tracking_params,
):
    """Track Gemini embeddings generation"""
    try:
        effective_customer_id = get_effective_customer_id(customer_id)
        if not effective_customer_id:
            logger.warning("No customer_id provided for embeddings tracking")
            return

        # Embeddings don't typically report token usage like text generation
        # But we track the operation for billing and analytics
        model = kwargs.get("model", "unknown")
        if isinstance(model, str) and model.startswith("models/"):
            model = model[7:]

        metadata = {
            "operation": "embed_content",
            "content_length": len(kwargs.get("content", "")),
        }

        if result:
            # Add embedding-specific metadata if available
            if hasattr(result, "embedding") and hasattr(result.embedding, "values"):
                metadata["embedding_dimensions"] = len(result.embedding.values)

        if custom_metadata:
            metadata.update(custom_metadata)

        # For embeddings, we track as a custom event with estimated usage
        tracker.track_usage_background(
            customer_id=effective_customer_id,
            model=model,
            input_tokens=len(kwargs.get("content", "").split())
            if kwargs.get("content")
            else 0,  # Rough estimate
            output_tokens=0,  # Embeddings don't have output tokens
            provider="google",
            metadata=metadata,
            **tracking_params,
        )
    except Exception as e:
        logger.warning(f"Failed to track embeddings: {e}")


def track_batch_embed_contents(
    result,
    customer_id,
    tracker,
    method_name,
    args,
    kwargs,
    custom_metadata=None,
    **tracking_params,
):
    """Track Gemini batch embeddings generation"""
    try:
        effective_customer_id = get_effective_customer_id(customer_id)
        if not effective_customer_id:
            logger.warning("No customer_id provided for batch embeddings tracking")
            return

        model = kwargs.get("model", "unknown")
        if isinstance(model, str) and model.startswith("models/"):
            model = model[7:]

        requests = kwargs.get("requests", [])
        content_count = len(requests)
        total_content_length = sum(len(req.get("content", "")) for req in requests)

        metadata = {
            "operation": "batch_embed_contents",
            "batch_size": content_count,
            "total_content_length": total_content_length,
        }

        if result and hasattr(result, "embeddings"):
            metadata["embeddings_generated"] = len(result.embeddings)

        if custom_metadata:
            metadata.update(custom_metadata)

        # Estimate tokens based on total content
        estimated_tokens = total_content_length // 4  # Rough approximation

        tracker.track_usage_background(
            customer_id=effective_customer_id,
            model=model,
            input_tokens=estimated_tokens,
            output_tokens=0,
            provider="google",
            metadata=metadata,
            **tracking_params,
        )
    except Exception as e:
        logger.warning(f"Failed to track batch embeddings: {e}")


def track_classify_text(
    result,
    customer_id,
    tracker,
    method_name,
    args,
    kwargs,
    custom_metadata=None,
    **tracking_params,
):
    """Track Gemini text classification"""
    try:
        effective_customer_id = get_effective_customer_id(customer_id)
        if not effective_customer_id:
            logger.warning("No customer_id provided for text classification tracking")
            return

        model = kwargs.get("model", "unknown")
        if isinstance(model, str) and model.startswith("models/"):
            model = model[7:]

        text_content = kwargs.get("text", "")
        metadata = {
            "operation": "classify_text",
            "text_length": len(text_content),
        }

        if result:
            # Add classification results if available
            if hasattr(result, "categories"):
                metadata["categories_count"] = len(result.categories)
            if hasattr(result, "confidence"):
                metadata["confidence"] = result.confidence

        if custom_metadata:
            metadata.update(custom_metadata)

        # Estimate tokens for classification
        estimated_tokens = len(text_content.split()) * 1.3  # Rough token estimate

        tracker.track_usage_background(
            customer_id=effective_customer_id,
            model=model,
            input_tokens=int(estimated_tokens),
            output_tokens=0,  # Classification typically doesn't generate text
            provider="google",
            metadata=metadata,
            **tracking_params,
        )
    except Exception as e:
        logger.warning(f"Failed to track text classification: {e}")


def track_batch_generate_content(
    result,
    customer_id,
    tracker,
    method_name,
    args,
    kwargs,
    custom_metadata=None,
    **tracking_params,
):
    """Track Gemini batch content generation"""
    try:
        effective_customer_id = get_effective_customer_id(customer_id)
        if not effective_customer_id:
            logger.warning(
                "No customer_id provided for batch content generation tracking"
            )
            return

        model = kwargs.get("model", "unknown")
        if isinstance(model, str) and model.startswith("models/"):
            model = model[7:]

        requests = kwargs.get("requests", [])
        batch_size = len(requests)

        # Aggregate token usage if available in results
        total_input_tokens = 0
        total_output_tokens = 0

        if result and hasattr(result, "responses"):
            for response in result.responses:
                if hasattr(response, "usage_metadata"):
                    total_input_tokens += getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    )
                    total_output_tokens += getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    )

        metadata = {
            "operation": "batch_generate_content",
            "batch_size": batch_size,
            "total_requests": len(requests),
        }

        if custom_metadata:
            metadata.update(custom_metadata)

        tracker.track_usage_background(
            customer_id=effective_customer_id,
            model=model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            provider="google",
            metadata=metadata,
            **tracking_params,
        )
    except Exception as e:
        logger.warning(f"Failed to track batch content generation: {e}")


def track_start_chat(
    result,
    customer_id,
    tracker,
    method_name,
    args,
    kwargs,
    custom_metadata=None,
    **tracking_params,
):
    """Track Gemini chat session creation"""
    try:
        effective_customer_id = get_effective_customer_id(customer_id)
        if not effective_customer_id:
            logger.warning("No customer_id provided for chat session tracking")
            return

        model = kwargs.get("model", "unknown")
        if isinstance(model, str) and model.startswith("models/"):
            model = model[7:]

        history = kwargs.get("history", [])
        metadata = {
            "operation": "start_chat",
            "initial_history_length": len(history),
        }

        if result:
            # Track chat session ID if available
            if hasattr(result, "model"):
                metadata["chat_model"] = str(result.model)

        if custom_metadata:
            metadata.update(custom_metadata)

        # Chat creation itself doesn't consume tokens, but we track for analytics
        tracker.track_usage_background(
            customer_id=effective_customer_id,
            model=model,
            input_tokens=0,  # Chat session creation doesn't consume tokens
            output_tokens=0,
            provider="google",
            metadata=metadata,
            **tracking_params,
        )
    except Exception as e:
        logger.warning(f"Failed to track chat session creation: {e}")


# Google Gen AI tracking configuration - All methods that consume tokens or should be tracked
GEMINI_TRACK_METHODS = {
    # Text Generation
    "models.generate_content": track_generate_content,
    # Batch Generation
    "models.batch_generate_content": track_batch_generate_content,
    # Embeddings
    "models.embed_content": track_embed_content,
    "models.batch_embed_contents": track_batch_embed_contents,
    # Classification
    "models.classify_text": track_classify_text,
    # Chat Sessions
    "models.start_chat": track_start_chat,
    # Token Counting
    "models.count_tokens": track_count_tokens,
}
