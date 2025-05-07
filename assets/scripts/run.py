import os
import sys
import time
import webbrowser
import subprocess

def open_browser():
    time.sleep(2)
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(base_dir, "app.py")

    if not os.path.exists(app_path):
        print(f"App not found at: {app_path}")
        input("Press Enter to exit...")
        sys.exit(1)

    subprocess.Popen(["streamlit", "run", app_path, "--server.headless", "true"])

    open_browser()