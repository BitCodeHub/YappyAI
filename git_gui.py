#!/usr/bin/env python3
"""
Simple Git GUI for Claude Code
No terminal needed - just run this script
"""

import subprocess
import tkinter as tk
from tkinter import messagebox, scrolledtext
import os

class GitGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Git Push Helper 🚀")
        self.root.geometry("600x500")
        
        # Status display
        tk.Label(root, text="Git Status:", font=("Arial", 12, "bold")).pack(pady=5)
        self.status_text = scrolledtext.ScrolledText(root, height=10, width=70)
        self.status_text.pack(pady=5)
        
        # Commit message
        tk.Label(root, text="Commit Message:", font=("Arial", 12, "bold")).pack(pady=5)
        self.commit_entry = tk.Entry(root, width=70)
        self.commit_entry.pack(pady=5)
        self.commit_entry.insert(0, "feat: add web search functionality for current information queries")
        
        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="🔄 Refresh Status", command=self.refresh_status, 
                 bg="#4CAF50", fg="white", padx=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="📦 Add All Files", command=self.add_files,
                 bg="#2196F3", fg="white", padx=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="📝 Commit", command=self.commit,
                 bg="#FF9800", fg="white", padx=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="🚀 Push to GitHub", command=self.push,
                 bg="#9C27B0", fg="white", padx=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="⚡ Quick Push All", command=self.quick_push_all,
                 bg="#F44336", fg="white", padx=10).pack(side=tk.LEFT, padx=5)
        
        # Initial status
        self.refresh_status()
    
    def run_command(self, cmd):
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout + result.stderr
        except Exception as e:
            return f"Error: {str(e)}"
    
    def refresh_status(self):
        self.status_text.delete(1.0, tk.END)
        status = self.run_command("git status")
        self.status_text.insert(1.0, status)
    
    def add_files(self):
        result = self.run_command("git add .")
        messagebox.showinfo("Add Files", "All files added to staging area!")
        self.refresh_status()
    
    def commit(self):
        commit_msg = self.commit_entry.get()
        if not commit_msg:
            messagebox.showwarning("No Message", "Please enter a commit message!")
            return
        
        result = self.run_command(f'git commit -m "{commit_msg}"')
        messagebox.showinfo("Commit", f"Committed with message: {commit_msg}")
        self.refresh_status()
    
    def push(self):
        result = self.run_command("git push origin main")
        if "Everything up-to-date" in result or "To https://" in result:
            messagebox.showinfo("Push Complete", "Successfully pushed to GitHub! ✅")
        else:
            messagebox.showwarning("Push Result", result)
        self.refresh_status()
    
    def quick_push_all(self):
        # Add all files
        self.run_command("git add .")
        
        # Commit
        commit_msg = self.commit_entry.get()
        self.run_command(f'git commit -m "{commit_msg}"')
        
        # Push
        result = self.run_command("git push origin main")
        
        messagebox.showinfo("Quick Push Complete", 
                          f"All changes committed and pushed! ✅\nMessage: {commit_msg}")
        self.refresh_status()

if __name__ == "__main__":
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    root = tk.Tk()
    app = GitGUI(root)
    root.mainloop()