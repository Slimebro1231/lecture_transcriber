# Lecture Transcriber Project Goals

This project aims to create a robust system for transcribing lecture audio and making it accessible for AI-driven analysis and Q&A.

## Core Objectives:

1.  **High-Accuracy Transcription**: Convert "scuffed" lecture audio into clean, accurate text.
2.  **Two-Pass Transcription System**:
    *   **Streaming Pass**: Provide real-time, scrolling "lyrics" or subtitles during the lecture using a fast, smaller model.
    *   **Refining Pass**: Generate a highly accurate, final transcript after the lecture using a larger, more powerful model.
3.  **AI-Friendly Output**: Save all final transcripts as plain text (.txt) or Markdown (.md) files for easy consumption by local or cloud-hosted AIs. This format ensures minimal tool calls and complexity for AI integration.
4.  **RAG Integration (Future)**: Enable a lightweight local LLM to perform Q&A based on the transcribed lecture content using Retrieval-Augmented Generation (RAG) to manage token limits effectively.
5.  **Resumable Transcription**: Implement the ability to continue transcription from a specific timestamp or existing partial transcript.

## Key Technologies:

*   **Speech-to-Text**: `whisper.cpp` (for both streaming and refining passes, leveraging its performance on Apple Silicon).
*   **Local LLM**: To be determined (e.g., Mistral, Llama, Phi-3 via Ollama).

This document serves as a guide for any AI interacting with or contributing to this project.