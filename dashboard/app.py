"""Compatibility entry point for Streamlit Community Cloud.
The maintained dashboard lives in app/app.py.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"app"))
exec((ROOT/"app"/"app.py").read_text(encoding="utf-8"),globals())
