# PDF Intelligence Report Summarizer

## Overview

A desktop application for summarizing large PDF military intelligence reports (100-1000 pages) using both offline and online AI capabilities. The application provides a GUI interface built with Tkinter and can be packaged as a standalone Windows executable. It supports two summarization modes: offline extractive summarization using NLP libraries (NLTK/Sumy) and online AI-powered summarization using OpenAI's API.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Architecture
**Desktop GUI Application Pattern**: The system follows a traditional desktop application architecture with a Tkinter-based GUI as the presentation layer. The codebase is organized into specialized modules that handle distinct responsibilities:

- **main.py**: Entry point containing the GUI controller (PDFSummarizerApp class) that orchestrates user interactions and coordinates between processing modules
- **pdf_processor.py**: Handles PDF parsing and text extraction using pypdf library
- **summarizer.py**: Implements dual-mode summarization logic (offline and online)
- **exporter.py**: Manages output generation in multiple formats (TXT, PDF)
- **build_exe.py**: PyInstaller configuration for creating standalone executables

**Rationale**: This modular separation allows each component to be developed, tested, and maintained independently while keeping the GUI logic separate from business logic.

### Dual-Mode Summarization Strategy
**Hybrid AI Approach**: The application implements two distinct summarization engines:

1. **Offline Mode**: Uses extractive summarization via Sumy library (LSA and Luhn algorithms) with NLTK tokenization
   - Pros: No internet required, no API costs, privacy-preserving
   - Cons: Lower quality summaries, limited contextual understanding

2. **Online Mode**: Leverages OpenAI's API for generative summarization
   - Pros: Higher quality, context-aware summaries, better handling of complex documents
   - Cons: Requires internet, API costs, dependency on external service

**Rationale**: This dual-mode approach provides flexibility for different use cases - offline mode for sensitive/classified documents or air-gapped environments, online mode for better quality when connectivity and API access are available.

### Text Processing Pipeline
**Progressive Processing with Callback Pattern**: The system implements a multi-stage processing pipeline:

1. PDF text extraction with page-by-page processing
2. Text validation and preprocessing
3. Summarization with configurable detail levels (high: 30-40%, medium: 15-20%, low: 5-10%)
4. Export to formatted outputs

**Threading for UI Responsiveness**: Heavy processing tasks run in background threads with progress callbacks to prevent GUI freezing.

**Rationale**: The callback pattern allows real-time progress updates to users during long-running operations (important for 1000+ page documents), while threading ensures the UI remains responsive.

### Export and Output Generation
**Multi-Format Export**: Uses ReportLab for PDF generation with professional formatting including:
- Custom styling and typography
- Metadata embedding (timestamps, source document info)
- Structured layout with headers and spacing

**Rationale**: Providing both TXT and formatted PDF outputs accommodates different downstream use cases - TXT for further processing/analysis, PDF for presentation and archival.

### Executable Distribution
**PyInstaller Packaging**: The application uses PyInstaller with specific configurations:
- `--onefile`: Single executable bundle
- `--windowed`: No console window for GUI app
- Hidden imports for dynamic dependencies (NLTK, Sumy, OpenAI)
- Data collection for NLTK/Sumy resources

**Rationale**: Creating a standalone executable eliminates Python installation requirements and simplifies deployment to end users, particularly important for military/government contexts where software installation may be restricted.

## External Dependencies

### Core Libraries
- **pypdf**: PDF parsing and text extraction (handles encrypted PDFs, page-by-page extraction)
- **tkinter**: Native GUI framework (bundled with Python, cross-platform compatibility)
- **PyInstaller**: Executable packaging and distribution

### NLP and Summarization
- **NLTK (Natural Language Toolkit)**: Tokenization and text processing for offline mode
  - Requires: punkt tokenizer data (auto-downloaded on first run)
- **Sumy**: Extractive summarization library
  - Implements: LSA (Latent Semantic Analysis) and Luhn algorithms
  - Provides: Stemming and stop word filtering

### Online AI Services
- **OpenAI API**: Cloud-based generative summarization
  - Requires: API key from https://platform.openai.com
  - Usage: Only when online mode is selected by user

### Document Generation
- **ReportLab**: PDF creation and formatting
  - Handles: Professional document layout, custom styling, metadata embedding
  - Used for: Exporting formatted summary reports

### Constraints and Limitations
- **PDF Size Limits**: Maximum 500MB file size, maximum 2000 pages
- **Internet Dependency**: Online mode requires active internet connection and valid OpenAI API key
- **Platform**: Primary target is Windows (EXE build), though core application is cross-platform compatible
