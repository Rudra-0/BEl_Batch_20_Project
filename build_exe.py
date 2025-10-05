import PyInstaller.__main__
import os
import sys

print("Building PDF Summarizer EXE...")
print("This may take several minutes...")

build_args = [
    'main.py',
    '--name=PDF_Summarizer',
    '--onefile',
    '--windowed',
    '--hidden-import=nltk',
    '--hidden-import=nltk.tokenize',
    '--hidden-import=nltk.data',
    '--hidden-import=sumy',
    '--hidden-import=sumy.parsers.plaintext',
    '--hidden-import=sumy.nlp.tokenizers',
    '--hidden-import=sumy.summarizers.lsa',
    '--hidden-import=sumy.nlp.stemmers',
    '--hidden-import=pypdf',
    '--hidden-import=openai',
    '--hidden-import=reportlab',
    '--hidden-import=tkinter',
    '--collect-data=nltk',
    '--collect-data=sumy',
    '--collect-data=reportlab',
    '--noconfirm',
    '--clean',
]

PyInstaller.__main__.run(build_args)

print("\n" + "="*60)
print("Build complete!")
print("="*60)
print(f"\nYour EXE file is located at: dist/PDF_Summarizer.exe")
print("\nIMPORTANT NOTES:")
print("1. The EXE is standalone and can run on any Windows machine")
print("2. For OFFLINE mode: No internet required")
print("3. For ONLINE mode: Requires internet and OpenAI API key")
print("4. First run may take a few seconds to initialize")
print("\nYou can now distribute dist/PDF_Summarizer.exe to users!")
