"""Compatibility import; the HTTP implementation lives in api/app.py."""
import sys
from api import app as implementation
sys.modules[__name__] = implementation
