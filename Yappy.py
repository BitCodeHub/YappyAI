#!/usr/bin/env python3
"""
Yappy - One-Click Launcher
This script automatically starts everything needed for Yappy
"""

import subprocess
import sys
import os
import time
import webbrowser
import threading
import signal
import platform
from pathlib import Path

class YappyLauncher:
    def __init__(self):
        self.api_process = None
        self.frontend_process = None
        self.running = False
        
    def check_python(self):
        """Check if Python is properly installed"""
        try:
            version = sys.version_info
            if version.major < 3 or (version.major == 3 and version.minor < 8):
                print("❌ Python 3.8 or higher is required")
                return False
            print(f"✅ Python {version.major}.{version.minor} detected")
            return True
        except:
            print("❌ Python not detected")
            return False
    
    def install_dependencies(self):
        """Auto-install dependencies if needed"""
        try:
            import pypdf
            import matplotlib
            import pandas
            import fastapi
            print("✅ All dependencies already installed")
            return True
        except ImportError:
            print("📦 Installing dependencies (this may take a few minutes)...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("✅ Dependencies installed successfully")
                return True
            except:
                print("❌ Failed to install dependencies")
                print("Please run: pip install -r requirements.txt")
                return False
    
    def check_port(self, port):
        """Check if a port is available"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) != 0
    
    def find_free_port(self, start_port=8000):
        """Find a free port starting from start_port"""
        for port in range(start_port, start_port + 100):
            if self.check_port(port):
                return port
        return None
    
    def start_api(self, port=8000):
        """Start the API server"""
        if not self.check_port(port):
            print(f"⚠️  Port {port} is already in use")
            port = self.find_free_port(port)
            if not port:
                print("❌ No free ports available")
                return False
            print(f"📌 Using port {port} instead")
        
        print(f"🚀 Starting API server on port {port}...")
        
        # Start API in subprocess
        if platform.system() == "Windows":
            self.api_process = subprocess.Popen(
                [sys.executable, "api_clean.py", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            self.api_process = subprocess.Popen(
                [sys.executable, "api_clean.py", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        # Wait for API to start
        time.sleep(3)
        
        # Check if API started successfully
        if self.api_process.poll() is None:
            print("✅ API server started successfully")
            return True
        else:
            print("❌ API server failed to start")
            return False
    
    def open_browser(self, port=3000):
        """Open the web browser"""
        time.sleep(2)  # Give frontend time to start
        url = f"http://localhost:{port}"
        print(f"🌐 Opening browser at {url}")
        webbrowser.open(url)
    
    def monitor_processes(self):
        """Monitor and restart processes if they crash"""
        while self.running:
            if self.api_process and self.api_process.poll() is not None:
                print("⚠️  API server stopped unexpectedly, restarting...")
                self.start_api()
            time.sleep(5)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutting down Yappy...")
        self.shutdown()
        sys.exit(0)
    
    def shutdown(self):
        """Shutdown all processes"""
        self.running = False
        
        if self.api_process:
            print("Stopping API server...")
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=5)
            except:
                self.api_process.kill()
        
        print("✅ Yappy stopped")
    
    def run(self):
        """Main run method"""
        print("""
╔═══════════════════════════════════════╗
║         Yappy PDF Assistant           ║
║   One-Click Launcher - Version 1.0    ║
╚═══════════════════════════════════════╝
        """)
        
        # Check Python
        if not self.check_python():
            input("\nPress Enter to exit...")
            return
        
        # Install dependencies
        print("\n🔧 Checking dependencies...")
        if not self.install_dependencies():
            input("\nPress Enter to exit...")
            return
        
        # Start API
        print("\n🚀 Starting services...")
        if not self.start_api():
            input("\nPress Enter to exit...")
            return
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Start monitoring thread
        self.running = True
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        # Open browser
        self.open_browser()
        
        # Show status
        print("\n" + "="*50)
        print("✅ Yappy is running!")
        print("="*50)
        print("\n📄 You can now:")
        print("  • Upload PDF files for analysis")
        print("  • Ask questions about your documents")
        print("  • Upload CSV/Excel files for visualization")
        print("\n🛑 To stop: Press Ctrl+C or close this window")
        print("="*50)
        
        # Keep running
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        self.shutdown()

def main():
    launcher = YappyLauncher()
    launcher.run()

if __name__ == "__main__":
    main()