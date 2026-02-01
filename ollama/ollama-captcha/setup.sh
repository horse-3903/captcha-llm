#!/bin/bash

cd "$(dirname "$0")"
pip install -r requirements.txt
python -u src/generate_data.py

# python -u main.py