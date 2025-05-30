#!/usr/bin/env python3
"""
Create a standalone app bundle for Yappy
"""

import os
import shutil
import platform
import stat
import subprocess

def create_macos_app():
    """Create a macOS .app bundle"""
    print("🍎 Creating macOS app bundle...")
    
    app_name = "Yappy.app"
    if os.path.exists(app_name):
        shutil.rmtree(app_name)
    
    # Create app structure
    os.makedirs(f"{app_name}/Contents/MacOS")
    os.makedirs(f"{app_name}/Contents/Resources")
    
    # Create Info.plist
    info_plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Yappy</string>
    <key>CFBundleDisplayName</key>
    <string>Yappy PDF Assistant</string>
    <key>CFBundleIdentifier</key>
    <string>com.yappy.app</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.10</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>"""
    
    with open(f"{app_name}/Contents/Info.plist", "w") as f:
        f.write(info_plist)
    
    # Create launcher script
    launcher_script = """#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../../.."
/usr/bin/python3 Yappy.py
"""
    
    launcher_path = f"{app_name}/Contents/MacOS/launcher"
    with open(launcher_path, "w") as f:
        f.write(launcher_script)
    
    # Make executable
    st = os.stat(launcher_path)
    os.chmod(launcher_path, st.st_mode | stat.S_IEXEC)
    
    print(f"✅ Created {app_name}")
    print("   You can now double-click it to run Yappy!")

def create_windows_exe():
    """Create Windows executable (requires pyinstaller)"""
    print("🪟 Creating Windows executable...")
    
    # Create spec file for PyInstaller
    spec_content = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['Yappy.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('api_clean.py', '.'),
        ('requirements.txt', '.'),
        ('sources', 'sources'),
        ('prompts', 'prompts'),
        ('frontend', 'frontend'),
    ],
    hiddenimports=['pypdf', 'matplotlib', 'seaborn', 'pandas', 'fastapi', 'uvicorn'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Yappy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico'
)
"""
    
    with open("AgenticSeek.spec", "w") as f:
        f.write(spec_content)
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print("📦 Building executable with PyInstaller...")
        subprocess.run([sys.executable, "-m", "PyInstaller", "Yappy.spec", "--noconfirm"])
        print("✅ Created Yappy.exe in dist/ folder")
    except ImportError:
        print("⚠️  PyInstaller not installed")
        print("To create .exe file, run: pip install pyinstaller")
        print("Then run: pyinstaller Yappy.spec")

def create_desktop_shortcut():
    """Create desktop shortcut"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        desktop = os.path.expanduser("~/Desktop")
        shortcut_path = os.path.join(desktop, "Yappy")
        
        # Create alias/symlink
        if os.path.exists("Yappy.app"):
            os.symlink(os.path.abspath("Yappy.app"), shortcut_path)
            print(f"✅ Created desktop shortcut: {shortcut_path}")
    
    elif system == "Windows":
        # Create .lnk file (requires win32com)
        try:
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
            desktop = shell.SpecialFolders("Desktop")
            shortcut = shell.CreateShortCut(os.path.join(desktop, "Yappy.lnk"))
            shortcut.Targetpath = os.path.abspath("Yappy.bat")
            shortcut.WorkingDirectory = os.path.dirname(os.path.abspath("Yappy.bat"))
            shortcut.save()
            print(f"✅ Created desktop shortcut")
        except ImportError:
            print("ℹ️  To create desktop shortcut, drag Yappy.bat to your desktop")

def main():
    print("""
╔═══════════════════════════════════════╗
║       Yappy App Bundle Creator        ║
╚═══════════════════════════════════════╝
    """)
    
    system = platform.system()
    
    # Make scripts executable
    if system != "Windows":
        os.chmod("Yappy.py", 0o755)
        if os.path.exists("Yappy.command"):
            os.chmod("Yappy.command", 0o755)
    
    if system == "Darwin":  # macOS
        create_macos_app()
        create_desktop_shortcut()
        print("\n✅ Done! You can now:")
        print("1. Double-click 'Yappy.app' to run")
        print("2. Drag it to Applications folder")
        print("3. Or use the desktop shortcut")
        
    elif system == "Windows":
        create_windows_exe()
        create_desktop_shortcut()
        print("\n✅ Done! You can now:")
        print("1. Double-click 'Yappy.bat' to run")
        print("2. Or use 'Yappy.exe' (if created)")
        
    else:  # Linux
        # Create .desktop file
        desktop_file = """[Desktop Entry]
Type=Application
Name=Yappy
Comment=PDF Assistant with AI
Exec=python3 {}/Yappy.py
Path={}
Terminal=true
Icon=application-pdf
Categories=Office;Utility;
""".format(os.getcwd(), os.getcwd())
        
        desktop_path = os.path.expanduser("~/.local/share/applications/yappy.desktop")
        os.makedirs(os.path.dirname(desktop_path), exist_ok=True)
        
        with open(desktop_path, "w") as f:
            f.write(desktop_file)
        
        print(f"✅ Created desktop entry: {desktop_path}")
        print("You can now find Yappy in your applications menu")

if __name__ == "__main__":
    main()