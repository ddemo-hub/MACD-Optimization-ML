#!/bin/bash
sudo apt-get update
sudo apt-get install python3.10

python3.10 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt