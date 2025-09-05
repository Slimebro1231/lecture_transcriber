#!/usr/bin/env python3
"""
Memory monitoring script for the transcription system.
Run this alongside the main application to monitor memory usage.
"""

import psutil
import time
import sys
import os

def monitor_memory(pid=None, interval=5):
    """Monitor memory usage of a specific process or current process."""
    if pid is None:
        pid = os.getpid()
    
    print(f"🔍 Monitoring memory usage for PID {pid}")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 60)
    
    try:
        process = psutil.Process(pid)
        while True:
            try:
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = process.memory_percent()
                
                # Get system memory info
                system_memory = psutil.virtual_memory()
                
                print(f"📊 Memory: {memory_mb:.1f} MB ({memory_percent:.1f}%) | "
                      f"System: {system_memory.percent:.1f}% used | "
                      f"Available: {system_memory.available / 1024 / 1024 / 1024:.1f} GB")
                
                # Warning thresholds
                if memory_mb > 1000:
                    print(f"⚠️  WARNING: High memory usage ({memory_mb:.1f} MB)")
                if memory_mb > 2000:
                    print(f"🚨 CRITICAL: Very high memory usage ({memory_mb:.1f} MB)")
                if system_memory.percent > 90:
                    print(f"🚨 CRITICAL: System memory pressure ({system_memory.percent:.1f}%)")
                
                time.sleep(interval)
                
            except psutil.NoSuchProcess:
                print(f"❌ Process {pid} no longer exists")
                break
            except Exception as e:
                print(f"❌ Error monitoring process: {e}")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

def find_transcription_process():
    """Find the transcription process by name."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'gui_transcriber.py' in cmdline or 'engine.py' in cmdline:
                    return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def main():
    if len(sys.argv) > 1:
        try:
            pid = int(sys.argv[1])
            monitor_memory(pid)
        except ValueError:
            print("❌ Invalid PID. Please provide a numeric process ID.")
    else:
        # Try to find the transcription process automatically
        pid = find_transcription_process()
        if pid:
            print(f"🎯 Found transcription process: PID {pid}")
            monitor_memory(pid)
        else:
            print("❌ Could not find transcription process automatically.")
            print("Usage: python memory_monitor.py [PID]")
            print("Or run the transcription system first, then run this script.")

if __name__ == "__main__":
    main()
