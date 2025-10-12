#!/usr/bin/env python3
"""
Validation script to verify all dependencies are installed correctly
"""

import sys

print("PDF Intelligence Report Summarizer - Dependency Check")
print("=" * 60)

dependencies = {
    'pypdf': 'PDF processing',
    'openai': 'Online AI summarization',
    'nltk': 'Natural language processing',
    'sumy': 'Offline summarization',
    'reportlab': 'PDF export',
    'tkinter': 'GUI interface'
}

all_good = True

for module, description in dependencies.items():
    try:
        if module == 'tkinter':
            import tkinter
        else:
            __import__(module)
        print(f"✓ {module:15s} - {description}")
    except ImportError as e:
        print(f"✗ {module:15s} - MISSING: {description}")
        all_good = False

print("=" * 60)

if all_good:
    print("\n✓ All dependencies installed successfully!")
    print("\nYour application is ready to build into an EXE.")
    print("\nNext steps:")
    print("1. Download all files to your Windows computer")
    print("2. Run: python build_exe.py")
    print("3. Find your EXE in the dist/ folder")
else:
    print("\n✗ Some dependencies are missing.")
    print("Please install them before building the EXE.")

print("\nApplication modules check:")
try:
    from pdf_processor import PDFProcessor
    print("✓ pdf_processor.py - Ready")
except Exception as e:
    print(f"✗ pdf_processor.py - Error: {e}")

try:
    from summarizer import Summarizer
    print("✓ summarizer.py - Ready")
except Exception as e:
    print(f"✗ summarizer.py - Error: {e}")

try:
    from exporter import SummaryExporter
    print("✓ exporter.py - Ready")
except Exception as e:
    print(f"✗ exporter.py - Error: {e}")

print("\n" + "=" * 60)
print("Build file: build_exe.py is ready to use")
print("=" * 60)
