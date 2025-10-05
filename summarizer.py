import os
from typing import Optional, Literal
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

class Summarizer:
    def __init__(self, mode: Literal['offline', 'online'], api_key: Optional[str] = None):
        self.mode = mode
        self.api_key = api_key
        
        if mode == 'online':
            if not api_key:
                raise ValueError("API key required for online mode")
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
    
    def summarize(self, text: str, detail_level: Literal['high', 'medium', 'low'], 
                  progress_callback=None) -> str:
        """
        Summarize text with specified detail level
        - high: 30-40% of original
        - medium: 15-20% of original  
        - low: 5-10% of original
        """
        
        if self.mode == 'offline':
            return self._summarize_offline(text, detail_level, progress_callback)
        else:
            return self._summarize_online(text, detail_level, progress_callback)
    
    def _summarize_offline(self, text: str, detail_level: str, progress_callback=None) -> str:
        """Offline summarization using extractive methods"""
        try:
            if not text or not text.strip():
                raise ValueError("No text provided for summarization")
            
            if progress_callback:
                progress_callback(10, "Preparing text for summarization...")
            
            word_count = len(text.split())
            
            if word_count < 50:
                raise ValueError("Text is too short to summarize (minimum 50 words)")
            
            if detail_level == 'high':
                sentence_count = max(int(word_count * 0.35 / 20), 10)
            elif detail_level == 'medium':
                sentence_count = max(int(word_count * 0.175 / 20), 5)
            else:
                sentence_count = max(int(word_count * 0.075 / 20), 3)
            
            if progress_callback:
                progress_callback(30, "Analyzing text structure...")
            
            try:
                parser = PlaintextParser.from_string(text, Tokenizer("english"))
            except Exception as parse_error:
                raise Exception(f"Failed to parse text: {str(parse_error)}")
            
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")
            
            if progress_callback:
                progress_callback(60, "Generating summary...")
            
            summary = summarizer(parser.document, sentence_count)
            
            if progress_callback:
                progress_callback(90, "Finalizing summary...")
            
            if not summary:
                raise Exception("Summarization produced no output")
            
            summary_text = " ".join([str(sentence) for sentence in summary])
            
            if not summary_text or len(summary_text.strip()) < 20:
                raise Exception("Summary is too short or empty")
            
            if progress_callback:
                progress_callback(100, "Summary complete!")
            
            return summary_text
            
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise Exception(f"Error in offline summarization: {str(e)}")
    
    def _summarize_online(self, text: str, detail_level: str, progress_callback=None) -> str:
        """Online summarization using OpenAI API"""
        try:
            if not text or not text.strip():
                raise ValueError("No text provided for summarization")
            
            word_count = len(text.split())
            if word_count < 50:
                raise ValueError("Text is too short to summarize (minimum 50 words)")
            
            if progress_callback:
                progress_callback(20, "Connecting to AI service...")
            
            max_length = 120000
            if len(text) > max_length:
                if progress_callback:
                    progress_callback(25, f"Truncating text from {len(text)} to {max_length} characters...")
                text = text[:max_length]
            
            if detail_level == 'high':
                detail_instruction = "Provide a comprehensive summary covering all major points and important details."
            elif detail_level == 'medium':
                detail_instruction = "Provide a moderate summary covering main points and key details."
            else:
                detail_instruction = "Provide a brief summary covering only the most critical points."
            
            if progress_callback:
                progress_callback(40, "Sending request to AI...")
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {"role": "system", "content": "You are an expert at summarizing military intelligence reports. Maintain professionalism and accuracy."},
                        {"role": "user", "content": f"{detail_instruction}\n\nText to summarize:\n\n{text}"}
                    ],
                    temperature=0.3,
                    timeout=120
                )
            except Exception as api_error:
                error_msg = str(api_error)
                if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                    raise Exception("Invalid API key. Please check your OpenAI API key.")
                elif "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    raise Exception("OpenAI API quota exceeded or rate limited. Please try again later or use offline mode.")
                elif "timeout" in error_msg.lower():
                    raise Exception("Request timed out. The text may be too long. Try using offline mode.")
                else:
                    raise Exception(f"OpenAI API error: {error_msg}")
            
            if progress_callback:
                progress_callback(90, "Processing response...")
            
            if not response.choices or not response.choices[0].message.content:
                raise Exception("No response received from AI service")
            
            summary = response.choices[0].message.content
            
            if not summary or len(summary.strip()) < 20:
                raise Exception("AI service returned an empty or invalid summary")
            
            if progress_callback:
                progress_callback(100, "Summary complete!")
            
            return summary
            
        except ValueError as ve:
            raise ve
        except Exception as e:
            if "Error in online summarization" not in str(e):
                raise Exception(f"Error in online summarization: {str(e)}")
            raise e
