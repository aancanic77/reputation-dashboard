import os
import sys
import subprocess

# Add the "app" folder to Python path
APP_DIR = os.path.join(os.path.dirname(__file__), "app")
sys.path.append(APP_DIR)

# Optional: ensure working directory is "app"
os.chdir(APP_DIR)

# Launch the real Streamlit app inside the app/ folder
subprocess.run(["streamlit", "run", "streamlit_app.py"])
