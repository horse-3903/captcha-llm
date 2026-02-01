#!/bin/bash

cd "$(dirname "$0")"

zip -r ollama-captcha.zip . \
  -x "data/*" \
  -x "results/*" \
  -x "__pycache__/*" \
  -x "*/__pycache__/*"