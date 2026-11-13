import sys
from pathlib import Path

# src/ scripts import each other as top-level modules (e.g. `from utils...`)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
