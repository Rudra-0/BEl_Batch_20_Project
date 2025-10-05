import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
from pdf_processor import PDFProcessor
from summarizer import Summarizer
from exporter import SummaryExporter
from datetime import datetime

class PDFSummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Military Intelligence Report Summarizer")
        self.root.geometry("900x700")
        
        self.pdf_path = None
        self.extracted_text = ""
        self.summary = ""
        self.stats = None
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
        ttk.Label(main_frame, text="PDF Intelligence Report Summarizer", 
                 font=('Arial', 16, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        pdf_frame = ttk.LabelFrame(main_frame, text="1. Select PDF File", padding="10")
        pdf_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        pdf_frame.columnconfigure(1, weight=1)
        
        ttk.Button(pdf_frame, text="Browse PDF", 
                  command=self.browse_pdf).grid(row=0, column=0, padx=5)
        self.pdf_label = ttk.Label(pdf_frame, text="No file selected")
        self.pdf_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        mode_frame = ttk.LabelFrame(main_frame, text="2. Select AI Mode", padding="10")
        mode_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.mode_var = tk.StringVar(value="offline")
        ttk.Radiobutton(mode_frame, text="Offline AI (No internet required)", 
                       variable=self.mode_var, value="offline",
                       command=self.toggle_api_key).grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(mode_frame, text="Online AI (OpenAI - Better quality)", 
                       variable=self.mode_var, value="online",
                       command=self.toggle_api_key).grid(row=1, column=0, sticky=tk.W, padx=5)
        
        ttk.Label(mode_frame, text="OpenAI API Key:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=(5,0))
        self.api_key_entry = ttk.Entry(mode_frame, width=50, show="*")
        self.api_key_entry.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=5, pady=(0,5))
        self.api_key_entry.config(state='disabled')
        
        detail_frame = ttk.LabelFrame(main_frame, text="3. Select Summary Detail Level", padding="10")
        detail_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.detail_var = tk.StringVar(value="medium")
        ttk.Radiobutton(detail_frame, text="High Detail (30-40% of original)", 
                       variable=self.detail_var, value="high").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(detail_frame, text="Medium Detail (15-20% of original)", 
                       variable=self.detail_var, value="medium").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(detail_frame, text="Low Detail (5-10% of original)", 
                       variable=self.detail_var, value="low").grid(row=2, column=0, sticky=tk.W, padx=5)
        
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        self.process_btn = ttk.Button(action_frame, text="Process & Summarize", 
                                      command=self.process_pdf)
        self.process_btn.grid(row=0, column=0, padx=5)
        
        self.export_btn = ttk.Button(action_frame, text="Export Summary", 
                                     command=self.export_summary, state='disabled')
        self.export_btn.grid(row=0, column=1, padx=5)
        
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.grid(row=1, column=0, sticky=tk.W)
        
        output_frame = ttk.LabelFrame(main_frame, text="Summary Output", padding="10")
        output_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, 
                                                     height=15, font=('Arial', 10))
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def toggle_api_key(self):
        if self.mode_var.get() == "online":
            self.api_key_entry.config(state='normal')
        else:
            self.api_key_entry.config(state='disabled')
    
    def browse_pdf(self):
        filename = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.pdf_path = filename
            self.pdf_label.config(text=os.path.basename(filename))
    
    def update_progress(self, value, message):
        self.progress_bar['value'] = value
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def process_pdf(self):
        if not self.pdf_path:
            messagebox.showerror("Error", "Please select a PDF file first")
            return
        
        if self.mode_var.get() == "online" and not self.api_key_entry.get().strip():
            messagebox.showerror("Error", "Please enter your OpenAI API key for online mode")
            return
        
        self.process_btn.config(state='disabled')
        self.output_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self._process_thread)
        thread.daemon = True
        thread.start()
    
    def _process_thread(self):
        try:
            self.update_progress(0, "Starting PDF processing...")
            
            processor = PDFProcessor(self.pdf_path)
            self.extracted_text = processor.extract_text(
                progress_callback=lambda p, m: self.update_progress(p * 0.3, m)
            )
            
            self.stats = processor.get_text_stats()
            self.update_progress(30, f"Extracted {self.stats['pages']} pages, {self.stats['words']} words")
            
            api_key = self.api_key_entry.get().strip() if self.mode_var.get() == "online" else None
            summarizer = Summarizer(mode=self.mode_var.get(), api_key=api_key)
            
            self.summary = summarizer.summarize(
                self.extracted_text,
                self.detail_var.get(),
                progress_callback=lambda p, m: self.update_progress(30 + p * 0.7, m)
            )
            
            self.output_text.insert(1.0, f"=== SUMMARY ===\n")
            self.output_text.insert(tk.END, f"Mode: {self.mode_var.get().upper()}\n")
            self.output_text.insert(tk.END, f"Detail Level: {self.detail_var.get().upper()}\n")
            self.output_text.insert(tk.END, f"Original: {self.stats['words']} words, {self.stats['pages']} pages\n")
            self.output_text.insert(tk.END, f"Summary: {len(self.summary.split())} words\n")
            self.output_text.insert(tk.END, f"\n{'-'*60}\n\n")
            self.output_text.insert(tk.END, self.summary)
            
            self.update_progress(100, "Processing complete!")
            self.export_btn.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.update_progress(0, "Error occurred")
        finally:
            self.process_btn.config(state='normal')
    
    def export_summary(self):
        if not self.summary:
            messagebox.showwarning("Warning", "No summary to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Summary",
            defaultextension=".txt",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ],
            initialfile=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            try:
                file_ext = os.path.splitext(filename)[1].lower()
                
                if file_ext == '.pdf':
                    metadata = {
                        'AI Mode': self.mode_var.get().upper(),
                        'Detail Level': self.detail_var.get().upper(),
                        'Original Document': f"{self.stats['words']} words, {self.stats['pages']} pages",
                        'Summary Length': f"{len(self.summary.split())} words",
                        'Reduction': f"{100 - (len(self.summary.split())/self.stats['words']*100):.1f}%"
                    }
                    SummaryExporter.export_to_pdf(self.summary, filename, metadata)
                else:
                    content = self.output_text.get(1.0, tk.END)
                    SummaryExporter.export_to_txt(content, filename)
                
                messagebox.showinfo("Success", f"Summary exported to {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")

def main():
    root = tk.Tk()
    app = PDFSummarizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
