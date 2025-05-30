#!/usr/bin/env python3
"""
Stop any process using port 8000
"""

import subprocess
import sys
import os
import signal

def find_process_on_port(port):
    """Find process using the specified port"""
    try:
        # Use lsof to find process
        result = subprocess.run(['lsof', '-i', f':{port}'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Skip header
                # Parse PID from lsof output
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]  # PID is second column
        return None
    except Exception as e:
        print(f"Error finding process: {e}")
        return None

def kill_process(pid):
    """Kill process by PID"""
    try:
        os.kill(int(pid), signal.SIGTERM)
        print(f"✅ Killed process {pid}")
        return True
    except Exception as e:
        print(f"❌ Failed to kill process {pid}: {e}")
        return False

def main():
    print("🔍 Checking for processes on port 8000...")
    
    pid = find_process_on_port(8000)
    
    if pid:
        print(f"⚠️  Found process {pid} using port 8000")
        response = input("Do you want to kill this process? (y/n): ")
        
        if response.lower() == 'y':
            if kill_process(pid):
                print("\n✅ Port 8000 is now free!")
                print("\n🚀 You can now start the API with:")
                print("   python3 api_clean.py")
            else:
                print("\n❌ Failed to kill the process")
                print("Try running: sudo kill -9", pid)
        else:
            print("\n❌ Process not killed")
            print("The API cannot start while port 8000 is in use")
    else:
        print("✅ Port 8000 is free!")
        print("\n🚀 You can start the API with:")
        print("   python3 api_clean.py")

if __name__ == "__main__":
    main()