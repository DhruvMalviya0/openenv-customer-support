import uvicorn
import sys
import os

# Ensure the root directory is in the path so it can find main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

def main():
    """Entry point for the OpenEnv multi-mode deployment."""
    uvicorn.run("main:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()