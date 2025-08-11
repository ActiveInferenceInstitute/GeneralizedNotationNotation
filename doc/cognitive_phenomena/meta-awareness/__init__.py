import sys
from pathlib import Path

# Ensure local utils can be imported as 'utils' when running tests from repo root
_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

