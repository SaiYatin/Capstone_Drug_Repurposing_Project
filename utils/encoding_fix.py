"""Utility to fix Windows console encoding for Unicode output."""
import sys
import io


def fix_encoding():
    """Fix Windows cp1252 console encoding to support Unicode."""
    if sys.platform == 'win32':
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding='utf-8', errors='replace'
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding='utf-8', errors='replace'
            )
        except Exception:
            pass
