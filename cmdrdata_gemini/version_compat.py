"""
Version compatibility and detection for cmdrdata-gemini
"""

import sys
import warnings
from typing import Any, Dict, Optional, Tuple

try:
    from packaging import version
except ImportError:
    # Fallback version parsing
    class FakeVersion:
        def __init__(self, v):
            self.v = v

        def __lt__(self, other):
            return self.v < other.v

        def __le__(self, other):
            return self.v <= other.v

        def __gt__(self, other):
            return self.v > other.v

        def __ge__(self, other):
            return self.v >= other.v

        def __eq__(self, other):
            return self.v == other.v

        def __str__(self):
            return self.v

    def parse(v):
        return FakeVersion(v)

    version = type("Version", (), {"parse": parse})()


class VersionCompatibility:
    """Handles version detection and compatibility warnings for Google Gen AI"""

    # Supported Google Gen AI version ranges
    SUPPORTED_GENAI_VERSIONS = {
        "min": "0.1.0",
        "max": "1.0.0",  # Updated for latest versions
        "tested": [
            "0.1.0", "0.3.0", "0.5.0", "0.7.0", "0.8.0", 
            "0.9.0", "0.10.0", "0.11.0", "0.12.0", "0.13.0"
        ],
        "latest_tested": "0.13.0",
    }

    def __init__(self):
        self.genai_version = None
        self._check_genai_version()

    def _check_genai_version(self):
        """Check installed version of Google Gen AI SDK"""
        try:
            import google.genai as genai

            self.genai_version = genai.__version__
            self._validate_genai_version()
        except ImportError:
            warnings.warn(
                "Google Gen AI SDK not found. Please install it: pip install google-genai>=0.1.0",
                UserWarning,
                stacklevel=3,
            )

    def _validate_genai_version(self):
        """Validate Google Gen AI version and show warnings if needed"""
        if not self.genai_version:
            return

        current = version.parse(self.genai_version)
        min_version = version.parse(self.SUPPORTED_GENAI_VERSIONS["min"])
        max_version = version.parse(self.SUPPORTED_GENAI_VERSIONS["max"])

        if current < min_version:
            warnings.warn(
                f"cmdrdata-gemini: Google Gen AI SDK version {self.genai_version} is below minimum "
                f"supported version {self.SUPPORTED_GENAI_VERSIONS['min']}. "
                f"Please upgrade: pip install google-genai>={self.SUPPORTED_GENAI_VERSIONS['min']}",
                UserWarning,
                stacklevel=3,
            )
        elif current >= max_version:
            warnings.warn(
                f"cmdrdata-gemini: Google Gen AI SDK version {self.genai_version} is newer than tested version. "
                f"cmdrdata-gemini was tested up to version {self.SUPPORTED_GENAI_VERSIONS['latest_tested']}. "
                f"Functionality may be limited. Please check for cmdrdata-gemini updates.",
                UserWarning,
                stacklevel=3,
            )
        # Only warn for significantly older untested versions, not newer ones
        elif current < version.parse("0.8.0") and str(current) not in self.SUPPORTED_GENAI_VERSIONS["tested"]:
            warnings.warn(
                f"cmdrdata-gemini: Google Gen AI SDK version {self.genai_version} has not been fully tested. "
                f"Latest tested version: {self.SUPPORTED_GENAI_VERSIONS['latest_tested']}. "
                f"Consider upgrading for best compatibility.",
                UserWarning,
                stacklevel=3,
            )

    def is_genai_supported(self) -> bool:
        """Check if Google Gen AI version is supported"""
        if not self.genai_version:
            return False

        current = version.parse(self.genai_version)
        min_version = version.parse(self.SUPPORTED_GENAI_VERSIONS["min"])
        max_version = version.parse(self.SUPPORTED_GENAI_VERSIONS["max"])

        return min_version <= current < max_version

    def get_compatibility_info(self) -> Dict[str, Any]:
        """Get comprehensive compatibility information"""
        return {
            "google_genai": {
                "installed": self.genai_version,
                "supported": self.is_genai_supported(),
                "min_supported": self.SUPPORTED_GENAI_VERSIONS["min"],
                "max_supported": self.SUPPORTED_GENAI_VERSIONS["max"],
                "tested_versions": self.SUPPORTED_GENAI_VERSIONS["tested"],
            },
            "python": {
                "version": f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}",
                "supported": sys.version_info >= (3, 9),
            },
        }


# Global instance
_version_compat = VersionCompatibility()


def check_compatibility() -> bool:
    """
    Check if the current environment is compatible with cmdrdata-gemini.

    Returns:
        True if compatible, False otherwise
    """
    return _version_compat.is_genai_supported()


def get_compatibility_info() -> Dict[str, Any]:
    """
    Get detailed compatibility information.

    Returns:
        Dictionary with compatibility details
    """
    return _version_compat.get_compatibility_info()