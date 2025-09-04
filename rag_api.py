#!/usr/bin/env python3
"""
RAG Query System with Gemini API

Alternative implementation using Google's Gemini API for more advanced features.
Requires a free Gemini API key from https://makersuite.google.com/app/apikey
"""

import os
import sys
import json
import requests
from datetime import datetime
from pathlib import Path

class TranscriptRAGAPI:
    def __init__(self, transcripts_dir="transcripts", api_key=None):
        self.transcripts_dir = Path(transcripts_dir)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
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
    
    def create_context(self, query, max_context=30000):
        """Create context from relevant transcript sections."""
        # For API version, we need to be more careful about context length
        context_parts = []
        current_length = 0
        
        for transcript in self.transcripts:
            if current_length + transcript["length"] > max_context:
                break
            
            context_parts.append(f"=== Session {transcript['session_date']} ===\n{transcript['content']}\n")
            current_length += transcript["length"]
        
        return "\n".join(context_parts)
    
    def query_gemini_api(self, question, context):
        """Query Gemini API with the question and context."""
        if not self.api_key:
            return "❌ No Gemini API key found. Set GEMINI_API_KEY environment variable or pass api_key parameter."
        
        prompt = f"""You are an AI assistant helping with lecture transcript analysis. 

CONTEXT (Lecture Transcripts):
{context}

QUESTION: {question}

Please provide a comprehensive answer based on the lecture content above. If the information isn't available in the transcripts, please say so clearly."""

        try:
            headers = {
                "Content-Type": "application/json",
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                }
            }
            
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    return "❌ No response generated from Gemini API"
            else:
                return f"❌ API Error {response.status_code}: {response.text}"
                
        except requests.exceptions.Timeout:
            return "❌ API request timed out"
        except Exception as e:
            return f"❌ Error querying Gemini API: {e}"
    
    def ask(self, question):
        """Main method to ask a question about the transcripts."""
        if not self.transcripts:
            return "❌ No transcripts available for querying."
        
        print(f"🤔 Question: {question}")
        print("📖 Creating context from transcripts...")
        
        context = self.create_context(question)
        print(f"📝 Context length: {len(context):,} characters")
        
        print("🧠 Querying Gemini API...")
        response = self.query_gemini_api(question, context)
        
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
        
        return self.query_gemini_api(question, context)

def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        print("📚 Lecture Transcript RAG System (API Version)")
        print("\nUsage:")
        print("  python rag_api.py 'your question here'")
        print("  python rag_api.py --list")
        print("  python rag_api.py --summary [session_index]")
        print("\nSetup:")
        print("  export GEMINI_API_KEY='your_api_key_here'")
        print("  # Get free API key from: https://makersuite.google.com/app/apikey")
        print("\nExamples:")
        print("  python rag_api.py 'What are the main topics discussed?'")
        print("  python rag_api.py 'Explain the concept of sampling error'")
        print("  python rag_api.py --list")
        print("  python rag_api.py --summary 0")
        return
    
    rag = TranscriptRAGAPI()
    
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
