# streamlit_app.py - Compatibility Redirect Layer
# Redirects context to the modular production frontend/app.py
import sys
import os

# Ensure the root directory resides in the module path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# Import the main frontend app which executes the Streamlit UI components
import frontend.app
