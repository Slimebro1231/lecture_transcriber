#!/usr/bin/env python3
"""
RAG Query System for Lecture Transcripts

This tool integrates with Gemini CLI to provide intelligent querying
of all session transcripts using RAG (Retrieval-Augmented Generation).
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

class TranscriptRAG:
    def __init__(self, transcripts_dir="transcripts"):
        self.transcripts_dir = Path(transcripts_dir)
        self.transcripts = []
        self.load_all_transcripts()
    
    def load_all_transcripts(self):
        """Load all session transcripts from the transcripts directory."""
        if not self.transcripts_dir.exists():
            print(f"❌ Transcripts directory not found: {self.transcripts_dir}")
            return
        
        transcript_files = list(self.transcripts_dir.glob("session_*.txt"))
        if not transcript_files:
            print(f"❌ No session transcripts found in {self.transcripts_dir}")
            return
        
        print(f"📚 Loading {len(transcript_files)} session transcripts...")
        
        for file_path in sorted(transcript_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract session info from filename
                filename = file_path.stem
                session_date = filename.replace("session_", "").replace("_", " ")[:8]  # YYYYMMDD
                
                self.transcripts.append({
                    "file": str(file_path),
                    "session_date": session_date,
                    "content": content,
                    "length": len(content)
                })
                
            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
        
        total_content = sum(t["length"] for t in self.transcripts)
        print(f"✅ Loaded {len(self.transcripts)} transcripts ({total_content:,} characters)")
    
    def create_context(self, query, max_context=50000):
        """Create context from relevant transcript sections."""
        # For now, we'll use all transcripts as context
        # In a more sophisticated system, we'd do semantic search
        context_parts = []
        current_length = 0
        
        for transcript in self.transcripts:
            if current_length + transcript["length"] > max_context:
                break
            
            context_parts.append(f"=== Session {transcript['session_date']} ===\n{transcript['content']}\n")
            current_length += transcript["length"]
        
        return "\n".join(context_parts)
    
    def query_gemini(self, question, context):
        """Query Gemini CLI with the question and context."""
        # Create a prompt that includes the context
        prompt = f"""You are an AI assistant helping with lecture transcript analysis. 

CONTEXT (Lecture Transcripts):
{context}

QUESTION: {question}

Please provide a comprehensive answer based on the lecture content above. If the information isn't available in the transcripts, please say so clearly."""

        try:
            # Use Gemini CLI with the correct syntax
            result = subprocess.run(
                ["gemini", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"❌ Gemini CLI error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "❌ Query timed out (60 seconds)"
        except Exception as e:
            return f"❌ Error querying Gemini: {e}"
    
    def ask(self, question):
        """Main method to ask a question about the transcripts."""
        if not self.transcripts:
            return "❌ No transcripts available for querying."
        
        print(f"🤔 Question: {question}")
        print("📖 Creating context from transcripts...")
        
        context = self.create_context(question)
        print(f"📝 Context length: {len(context):,} characters")
        
        print("🧠 Querying Gemini...")
        response = self.query_gemini(question, context)
        
        return response
    
    def list_sessions(self):
        """List all available sessions."""
        if not self.transcripts:
            print("❌ No transcripts available.")
            return
        
        print("📚 Available Sessions:")
        for i, transcript in enumerate(self.transcripts, 1):
            print(f"  {i}. {transcript['session_date']} ({transcript['length']:,} chars)")
    
    def get_session_summary(self, session_index=None):
        """Get a summary of a specific session or all sessions."""
        if session_index is not None:
            if 0 <= session_index < len(self.transcripts):
                transcript = self.transcripts[session_index]
                question = f"Provide a detailed summary of this lecture session from {transcript['session_date']}"
                context = transcript['content']
            else:
                return "❌ Invalid session index."
        else:
            question = "Provide a comprehensive summary of all the lecture sessions, highlighting key topics and themes."
            context = self.create_context(question)
        
        return self.query_gemini(question, context)

def main():
    """Main CLI interface."""
    rag = TranscriptRAG()
    
    if len(sys.argv) < 2:
        print("📚 Lecture Transcript RAG System")
        print("\nUsage:")
        print("  python rag_query.py 'your question here'")
        print("  python rag_query.py --list")
        print("  python rag_query.py --summary [session_index]")
        print("\nExamples:")
        print("  python rag_query.py 'What are the main topics discussed?'")
        print("  python rag_query.py 'Explain the concept of sampling error'")
        print("  python rag_query.py --list")
        print("  python rag_query.py --summary 0")
        return
    
    if sys.argv[1] == "--list":
        rag.list_sessions()
    elif sys.argv[1] == "--summary":
        session_index = int(sys.argv[2]) if len(sys.argv) > 2 else None
        response = rag.get_session_summary(session_index)
        print(f"\n📋 Summary:\n{response}")
    else:
        question = " ".join(sys.argv[1:])
        response = rag.ask(question)
        print(f"\n💡 Answer:\n{response}")

if __name__ == "__main__":
    main()
