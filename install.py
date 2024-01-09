import os
import subprocess  
import sys

file_path = os.path.realpath(__file__)

requirements = os.path.join(os.path.dirname(file_path), "requirements.txt")

try:
    subprocess.run(["pip", "install", "-r", requirements], check=True)
except subprocess.CalledProcessError as e:
    print(f"Failed to install requirements: {e.stderr.decode('utf-8')}")
    sys.exit(1)