"""
Tests for Google Gen AI version compatibility detection
"""

import sys
import warnings
from unittest.mock import Mock, patch

import pytest

from cmdrdata_gemini.version_compat import (
    VersionCompatibility,
    check_compatibility,
    get_compatibility_info,
)


class TestVersionCompatibility:
    def test_genai_version_detection(self):
        """Test detection of installed Google Gen AI version"""
        compat = VersionCompatibility()

        # Should detect some version (or warn if not installed)
        assert compat.genai_version is not None or len(warnings.filters) > 0

    def test_supported_genai_version(self):
        """Test that supported versions are marked as compatible"""
        with patch("cmdrdata_gemini.version_compat.version") as mock_version:
            # Mock a supported version (0.5.0 is between 0.1.0 and 1.0.0)
            mock_parse = Mock()
            mock_current = Mock()
            mock_min = Mock()
            mock_max = Mock()

            # Set up comparison results for 0.5.0 being between 0.1.0 and 1.0.0
            mock_current.__le__ = Mock(return_value=True)  # 0.5.0 <= 0.5.0
            mock_current.__lt__ = Mock(return_value=True)  # 0.5.0 < 1.0.0
            mock_min.__le__ = Mock(return_value=True)  # 0.1.0 <= 0.5.0
            mock_max.__lt__ = Mock(return_value=False)  # 1.0.0 < 0.5.0 (False)

            def parse_side_effect(v):
                if v == "0.5.0":
                    return mock_current
                elif v == "0.1.0":
                    return mock_min
                elif v == "1.0.0":
                    return mock_max
                return Mock()

            mock_parse.side_effect = parse_side_effect
            mock_version.parse = mock_parse

            with patch("google.genai.__version__", "0.5.0"):
                compat = VersionCompatibility()
                assert compat.is_genai_supported()

    def test_unsupported_genai_version(self):
        """Test handling of unsupported Google Gen AI versions"""
        with patch("cmdrdata_gemini.version_compat.version") as mock_version:
            # Mock an unsupported (too old) version
            mock_parse = Mock()
            mock_old_version = Mock()
            mock_old_version.__lt__ = Mock(return_value=True)
            mock_old_version.__ge__ = Mock(return_value=False)
            mock_parse.return_value = mock_old_version
            mock_version.parse = mock_parse

            with patch("google.genai.__version__", "0.0.1"):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    compat = VersionCompatibility()
                    assert not compat.is_genai_supported()
                    assert len(w) > 0
                    assert "below minimum" in str(w[0].message)

    def test_missing_genai(self):
        """Test handling when Google Gen AI SDK is not installed"""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "google.genai":
                raise ImportError("No module named 'google.genai'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                compat = VersionCompatibility()
                assert not compat.is_genai_supported()
                assert len(w) > 0
                assert "not found" in str(w[0].message)

    def test_version_warnings(self):
        """Test version compatibility warnings"""
        with patch("cmdrdata_gemini.version_compat.version") as mock_version:
            # Mock a newer untested version
            mock_parse = Mock()
            mock_new_version = Mock()
            mock_new_version.__lt__ = Mock(return_value=False)
            mock_new_version.__ge__ = Mock(return_value=True)
            mock_new_version.__str__ = Mock(return_value="0.99.0")
            mock_parse.return_value = mock_new_version
            mock_version.parse = mock_parse

            with patch("google.genai.__version__", "0.99.0"):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    compat = VersionCompatibility()
                    assert len(w) > 0
                    assert "newer than tested" in str(w[0].message)

    def test_get_compatibility_info(self):
        """Test compatibility information retrieval"""
        info = get_compatibility_info()

        assert "google_genai" in info
        assert "python" in info
        assert "version" in info["python"]
        assert info["python"]["supported"] == (sys.version_info >= (3, 9))

    def test_check_compatibility_function(self):
        """Test standalone compatibility check function"""
        result = check_compatibility()
        assert isinstance(result, bool)
