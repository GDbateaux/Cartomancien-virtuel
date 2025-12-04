import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / 'src'

sys.path.append(str(ROOT))
sys.path.append(str(SRC))
