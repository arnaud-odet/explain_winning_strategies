#!/bin/bash
# run.sh

# Set Python flags to ignore warnings
export PYTHONWARNINGS=ignore

# Run your script with stderr redirected
python -W ignore data_boxscore/main.py 2>/dev/null
