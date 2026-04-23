"""
streamlit_app.py
----------------
Thin entry point – keeps the original `streamlit run streamlit_app.py` command working.

All UI logic lives in src/ui/app.py.
"""

import runpy

runpy.run_path("src/ui/app.py")
