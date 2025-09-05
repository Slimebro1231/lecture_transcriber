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
    
    # Check for lightweight mode argument
    if [ "$1" = "--lightweight" ] || [ "$1" = "-l" ]; then
        echo "Starting lightweight transcription system..."
        python gui_lightweight.py
    else
        echo "Starting full transcription system..."
        echo "Use --lightweight or -l for memory-optimized version"
        python gui_transcriber.py
    fi
else
    echo "Failed to activate micromamba environment."
    exit 1
fi
