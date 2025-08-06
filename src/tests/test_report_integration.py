#!/usr/bin/env python3
"""
Test Report Integration Tests

This file contains tests migrated from test_report_comprehensive.py.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *

