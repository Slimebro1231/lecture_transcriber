# RAG Query System Usage

The RAG (Retrieval-Augmented Generation) system allows you to ask intelligent questions about all your lecture transcripts using Gemini CLI.

## Quick Start

### Simple Questions
```bash
python ask.py "What are the main topics discussed?"
python ask.py "Explain sampling error"
python ask.py "What is the Canadian Labour Force Survey?"
```

### Advanced Usage
```bash
# List all available sessions
python rag_query.py --list

# Get summary of all sessions
python rag_query.py --summary

# Get summary of specific session
python rag_query.py --summary 0

# Ask detailed questions
python rag_query.py "How does sample size affect sampling error?"
```

## Features

✅ **Automatic Transcript Loading**: Loads all session files from `transcripts/` folder
✅ **Context-Aware Queries**: Provides relevant transcript context to Gemini
✅ **Session Management**: Lists and summarizes individual sessions
✅ **Gemini CLI Integration**: Uses your existing Gemini CLI installation
✅ **Simple Interface**: Easy-to-use command-line tools

## How It Works

1. **Transcript Loading**: Automatically finds and loads all `session_*.txt` files
2. **Context Creation**: Creates relevant context from transcript content
3. **Gemini Query**: Sends question + context to Gemini CLI
4. **Intelligent Response**: Returns comprehensive answers based on lecture content

## Example Queries

- "What are the main statistical concepts discussed?"
- "Explain the relationship between sample size and sampling error"
- "What is the Canadian Labour Force Survey used for?"
- "What are the different types of employment status in the survey?"
- "How does economics relate to market interactions?"
- "What is the Milton Friedman pencil story about?"

## Files

- `rag_query.py` - Main RAG system with full features
- `ask.py` - Simple wrapper for quick questions
- `rag_api.py` - Alternative version using Gemini API (requires API key)

## Requirements

- Gemini CLI installed and configured
- Session transcripts in `transcripts/` folder
- Python 3.9+ with standard libraries

The system is now ready to answer questions about your lecture content!
