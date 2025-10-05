from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime
import os

class SummaryExporter:
    @staticmethod
    def export_to_txt(content: str, filepath: str) -> None:
        """Export summary to text file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            raise Exception(f"Failed to export TXT: {str(e)}")
    
    @staticmethod
    def export_to_pdf(content: str, filepath: str, metadata: dict = None) -> None:
        """Export summary to PDF file with proper formatting"""
        try:
            doc = SimpleDocTemplate(
                filepath,
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=1*inch,
                bottomMargin=0.75*inch
            )
            
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                textColor='#000000',
                spaceAfter=12,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=12,
                textColor='#000000',
                spaceAfter=6,
                spaceBefore=12,
                fontName='Helvetica-Bold'
            )
            
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['BodyText'],
                fontSize=11,
                leading=14,
                alignment=TA_LEFT,
                fontName='Helvetica'
            )
            
            story = []
            
            title = Paragraph("Military Intelligence Report Summary", title_style)
            story.append(title)
            story.append(Spacer(1, 0.3*inch))
            
            if metadata:
                meta_heading = Paragraph("Summary Information", heading_style)
                story.append(meta_heading)
                
                for key, value in metadata.items():
                    meta_text = Paragraph(f"<b>{key}:</b> {value}", body_style)
                    story.append(meta_text)
                
                story.append(Spacer(1, 0.2*inch))
                story.append(Paragraph("-" * 80, body_style))
                story.append(Spacer(1, 0.2*inch))
            
            summary_heading = Paragraph("Summary Content", heading_style)
            story.append(summary_heading)
            
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    cleaned_para = para.replace('\n', ' ').strip()
                    p = Paragraph(cleaned_para, body_style)
                    story.append(p)
                    story.append(Spacer(1, 0.15*inch))
            
            footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            footer = Paragraph(footer_text, styles['Normal'])
            story.append(Spacer(1, 0.3*inch))
            story.append(footer)
            
            doc.build(story)
            
        except Exception as e:
            raise Exception(f"Failed to export PDF: {str(e)}")
