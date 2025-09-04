#!/bin/bash

# This script activates the micromamba environment and starts the transcription GUI.
# It ensures that the application runs with the correct Python interpreter and dependencies.

# Path to your micromamba executable
MICROMAMBA_PATH="/Users/max/.micromamba/bin/micromamba"

# Activate the environment and run the Python script
eval "$($MICROMAMBA_PATH shell hook --shell=bash)"
micromamba activate lecture-transcriber

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "Micromamba environment activated."
    # Run the GUI application
    python gui_transcriber.py
else
    echo "Failed to activate micromamba environment."
    exit 1
fi
