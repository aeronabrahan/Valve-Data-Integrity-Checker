import os
import sys
import time
import webbrowser
import subprocess
import signal

def open_browser():
    """Open the default web browser after a short delay."""
    time.sleep(2)
    webbrowser.open("http://localhost:8501")

def main():
    # Determine the correct path to app.py
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(base_dir, "app.py")

    if not os.path.exists(app_path):
        print(f"App not found at: {app_path}")
        input("Press Enter to exit...")
        sys.exit(1)

    try:
        # Start Streamlit in headless mode
        streamlit_process = subprocess.Popen([
            "streamlit", "run", app_path, "--server.headless", "true"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        open_browser()

        # Wait for the Streamlit process to complete (or allow Ctrl+C to stop)
        streamlit_process.wait()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down...")
        streamlit_process.send_signal(signal.SIGINT)
        streamlit_process.terminate()
        streamlit_process.wait()
        sys.exit(0)

if __name__ == "__main__":
    main()
