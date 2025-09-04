#!/usr/bin/env python3
"""
Simple wrapper for the RAG query system.
Just run: python ask.py "your question here"
"""

import sys
from rag_query import TranscriptRAG

def main():
    if len(sys.argv) < 2:
        print("📚 Lecture Transcript Q&A System")
        print("\nUsage: python ask.py 'your question here'")
        print("\nExamples:")
        print("  python ask.py 'What are the main topics?'")
        print("  python ask.py 'Explain sampling error'")
        print("  python ask.py 'What is the Canadian Labour Force Survey?'")
        return
    
    question = " ".join(sys.argv[1:])
    rag = TranscriptRAG()
    response = rag.ask(question)
    print(f"\n💡 Answer:\n{response}")

if __name__ == "__main__":
    main()
