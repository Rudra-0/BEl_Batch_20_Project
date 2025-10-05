import pypdf
import os
from typing import Optional

class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.text = ""
        self.page_count = 0
        
    def extract_text(self, progress_callback=None) -> str:
        """Extract text from PDF file with progress updates"""
        try:
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
            file_size = os.path.getsize(self.pdf_path)
            if file_size == 0:
                raise ValueError("PDF file is empty")
            
            if file_size > 500 * 1024 * 1024:
                raise ValueError("PDF file is too large (max 500MB)")
            
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                if pdf_reader.is_encrypted:
                    raise ValueError("PDF is password-protected. Please decrypt it first.")
                
                self.page_count = len(pdf_reader.pages)
                
                if self.page_count == 0:
                    raise ValueError("PDF has no pages")
                
                if self.page_count > 2000:
                    raise ValueError(f"PDF has too many pages ({self.page_count}). Maximum is 2000.")
                
                extracted_text = []
                
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            extracted_text.append(text)
                    except Exception as page_error:
                        print(f"Warning: Could not extract text from page {i+1}: {page_error}")
                    
                    if progress_callback:
                        progress = int((i + 1) / self.page_count * 100)
                        progress_callback(progress, f"Extracting page {i+1}/{self.page_count}")
                
                if not extracted_text:
                    raise ValueError("No text could be extracted from PDF. It may be image-based or corrupted.")
                
                self.text = "\n\n".join(extracted_text)
                return self.text
                
        except pypdf.errors.PdfReadError as e:
            raise Exception(f"Invalid or corrupted PDF file: {str(e)}")
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def get_text_stats(self) -> dict:
        """Get statistics about the extracted text"""
        words = self.text.split()
        return {
            'pages': self.page_count,
            'characters': len(self.text),
            'words': len(words),
            'estimated_reading_time_minutes': len(words) / 200
        }
