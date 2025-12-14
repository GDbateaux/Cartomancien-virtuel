from pathlib import Path
import sys

def project_root():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS).parent
    return Path(__file__).parent.parent
