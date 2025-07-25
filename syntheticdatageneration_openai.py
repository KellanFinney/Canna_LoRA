from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore 
import os
import json
import logging
import time
from typing import List 
from pydantic import BaseModel
from openai import OpenAI
from generated_prompt import prompt_template
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
from datetime import datetime, timedelta
from pathlib import Path
from config_loader import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

# Rate limiting for OpenAI API
class RateLimiter:
    def __init__(self, max_requests_per_minute=490):  # Close to 500 RPM limit for maximum speed
        self.max_requests = max_requests_per_minute
        self.requests = Queue()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = datetime.now()
            
            # Remove requests older than 1 minute
            while not self.requests.empty():
                if now - self.requests.queue[0] > timedelta(minutes=1):
                    self.requests.get()
                else:
                    break
            
            # If we're at the limit, wait
            if self.requests.qsize() >= self.max_requests:
                sleep_time = 60 - (now - self.requests.queue[0]).total_seconds()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # Clean up old requests after waiting
                    now = datetime.now()
                    while not self.requests.empty() and now - self.requests.queue[0] > timedelta(minutes=1):
                        self.requests.get()
            
            # Record this request
            self.requests.put(now)

# Global rate limiter
rate_limiter = RateLimiter()

# OpenAI client setup
client = None

def get_client():
    global client
    if client is None:
        # Get API key from environment variable for security
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Get your API key from: https://platform.openai.com/api-keys"
            )
        
        client = OpenAI(api_key=api_key)
    return client

def llm_call(data: str, num_records: int = 5, model: str = "gpt-4o-mini") -> dict:
    """
    Generate Q&A pairs from text data using OpenAI's models
    
    Args:
        data: Text data to generate Q&A from
        num_records: Number of Q&A pairs to generate
        model: Model to use - options:
               - "gpt-4o-mini" (Recommended: fast, cheap, high quality)
               - "gpt-4o" (Best quality, more expensive)
               - "gpt-3.5-turbo" (Cheapest, still good)
    """
    # Apply rate limiting
    rate_limiter.wait_if_needed()
    
    try:
        response = get_client().chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt_template(data, num_records)
                }
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}  # Ensures JSON response
        )
        
        # Extract content from response
        content = response.choices[0].message.content
        
        # Parse JSON response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback parsing for any formatting issues
            import re
            
            # Try to find JSON in the response
            json_match = re.search(r'(\{.*?\})', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            return {"generated": []}
        
    except Exception as e:
        return {"generated": []}

# Global instances to reuse (much faster than creating new ones each time)
converter = None
chunker = None

def get_converter():
    global converter
    if converter is None:
        converter = DocumentConverter()
    return converter

def get_chunker():
    global chunker
    if chunker is None:
        chunker = HybridChunker()
    return chunker

def process_chunk(chunk_data):
    """Process a single chunk - designed for parallel execution"""
    chunk_idx, enriched_text, model, pdf_name = chunk_data  # Now receives pre-contextualized text
    
    try:
        # Generate Q&A pairs (no contextualization needed - already done)
        data = llm_call(enriched_text, num_records=5, model=model)
        
        if data and "generated" in data:
            return {
                "chunk_idx": chunk_idx,
                "success": True,
                "data": {
                    "generated": data["generated"], 
                    "context": enriched_text,
                    "source_pdf": pdf_name
                },
                "qa_count": len(data["generated"])
            }
        else:
            return {"chunk_idx": chunk_idx, "success": False, "qa_count": 0}
            
    except Exception as e:
        return {"chunk_idx": chunk_idx, "success": False, "error": str(e), "qa_count": 0}

def process_pdf(pdf_path: str, model: str = "gpt-4o-mini", progress_info: str = "", max_workers: int = 20) -> dict:
    """Process a single PDF and generate Q&A pairs using parallel processing"""
    
    try:
        # Use global instances (much faster)
        doc = get_converter().convert(pdf_path).document
        chunker = get_chunker()
        chunks = list(chunker.chunk(dl_doc=doc))
        
        pdf_name = Path(pdf_path).name
        
        print(f"\r{progress_info} - Contextualizing {len(chunks)} chunks...", end="", flush=True)
        
        # Pre-contextualize ALL chunks before API calls (faster batch operation)
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                enriched_text = chunker.contextualize(chunk=chunk)
                enriched_chunks.append((i, enriched_text, model, pdf_name))
                print(f"\r{progress_info} - Contextualizing {i+1}/{len(chunks)} chunks...", end="", flush=True)
            except Exception as e:
                print(f"\r{progress_info} - Error contextualizing chunk {i}: {str(e)[:30]}...", end="", flush=True)
                continue
        
        print(f"\r{progress_info} - Sending {len(enriched_chunks)} chunks to OpenAI...", end="", flush=True)
        
        pdf_dataset = {}
        total_qa_pairs = 0
        completed_chunks = 0
        
        # Now process enriched chunks in parallel (only API calls now)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(process_chunk, chunk_data): chunk_data[0] 
                for chunk_data in enriched_chunks
            }
            
            for future in as_completed(future_to_chunk):
                result = future.result()
                completed_chunks += 1
                
                print(f"\r{progress_info} - {completed_chunks}/{len(enriched_chunks)} API calls complete", end="", flush=True)
                
                if result["success"]:
                    chunk_idx = result["chunk_idx"]
                    pdf_dataset[f"chunk_{chunk_idx}"] = result["data"]
                    total_qa_pairs += result["qa_count"]
        
        print(f" ✓ ({total_qa_pairs} Q&A pairs)")
        return pdf_dataset
    
    except Exception as e:
        print(f" ✗ Error: {str(e)[:50]}...")
        return {}

def main():
    # Load configuration
    config = get_config()
    
    # Get settings from config
    PDF_DIRECTORY = config.data_generation.get('pdf_directory', './pdfs')
    OUTPUT_DIRECTORY = config.data_generation.get('output_directory', './data')
    PDFS_PER_BATCH = config.data_generation.get('pdfs_per_batch', 5)
    MAX_WORKERS = config.processing.get('max_workers', 30)
    MODEL = config.openai.get('model', 'gpt-4o-mini')
    
    # Convert to Path objects for cross-platform compatibility
    PDF_DIRECTORY = Path(PDF_DIRECTORY)
    OUTPUT_DIRECTORY = Path(OUTPUT_DIRECTORY)
    
    print(f"{Fore.CYAN}🚀 Starting OPTIMIZED Parallel Batch Synthetic Data Generation{Fore.RESET}")
    print(f"{Fore.CYAN}📁 PDF Directory: {PDF_DIRECTORY}{Fore.RESET}")
    print(f"{Fore.CYAN}💾 Output Directory: {OUTPUT_DIRECTORY}{Fore.RESET}")
    print(f"{Fore.CYAN}🤖 Model: {MODEL}{Fore.RESET}")
    print(f"{Fore.CYAN}📦 Batch Size: {PDFS_PER_BATCH} PDFs per JSON file{Fore.RESET}")
    print(f"{Fore.CYAN}⚡ Parallel Workers: {MAX_WORKERS} (Rate Limited to 490 RPM){Fore.RESET}")
    print(f"{Fore.GREEN}🔥 OPTIMIZATIONS: Reused instances + Batch contextualization + Increased workers{Fore.RESET}")
    
    # Pre-initialize global instances for maximum speed
    print(f"{Fore.YELLOW}🔧 Pre-initializing converter and chunker...{Fore.RESET}")
    get_converter()  # Initialize once
    get_chunker()    # Initialize once
    print(f"{Fore.GREEN}✅ Ready for maximum speed processing!{Fore.RESET}")
    
    # Check if directories exist
    if not os.path.exists(PDF_DIRECTORY):
        print(f"{Fore.RED}❌ Error: PDF directory {PDF_DIRECTORY} does not exist!{Fore.RESET}")
        return
    
    if not os.path.exists(OUTPUT_DIRECTORY):
        print(f"{Fore.YELLOW}📁 Creating output directory: {OUTPUT_DIRECTORY}{Fore.RESET}")
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # Get all PDF files using cross-platform paths
    pdf_files = [f.name for f in PDF_DIRECTORY.glob('*.pdf')]
    
    if not pdf_files:
        print(f"{Fore.RED}❌ No PDF files found in {PDF_DIRECTORY}{Fore.RESET}")
        return
    
    print(f"{Fore.GREEN}📄 Found {len(pdf_files)} PDF files{Fore.RESET}")
    
    # Calculate number of batches
    total_batches = (len(pdf_files) + PDFS_PER_BATCH - 1) // PDFS_PER_BATCH
    
    # Check for existing batches to resume from
    existing_batches = []
    for i in range(1, total_batches + 1):
        batch_file = OUTPUT_DIRECTORY / f"science_training_batch_{i:03d}.json"
        if batch_file.exists():
            existing_batches.append(i)
    
    start_batch = len(existing_batches)  # 0-indexed, so if 25 exist, start at batch 25 (26th batch)
    
    if existing_batches:
        print(f"{Fore.YELLOW}📋 Found {len(existing_batches)} existing batches - resuming from batch {start_batch + 1}{Fore.RESET}")
    
    print(f"{Fore.GREEN}📦 Will create {total_batches} JSON files total (processing {total_batches - start_batch} remaining){Fore.RESET}")
    print()
    
    # Process PDFs in batches (resume from where we left off)
    overall_stats = {"total_pdfs": 0, "total_chunks": 0, "total_qa_pairs": 0, "total_cost": 0}
    start_time = time.time()
    
    for batch_num in range(start_batch, total_batches):
        start_idx = batch_num * PDFS_PER_BATCH
        end_idx = min(start_idx + PDFS_PER_BATCH, len(pdf_files))
        batch_files = pdf_files[start_idx:end_idx]
        
        batch_start_time = time.time()
        print(f"{Fore.CYAN}📦 Batch {batch_num + 1}/{total_batches} - Processing {len(batch_files)} PDFs{Fore.RESET}")
        
        batch_dataset = {}
        batch_stats = {"chunks": 0, "qa_pairs": 0}
        
        for pdf_idx, pdf_file in enumerate(batch_files):
            pdf_path = PDF_DIRECTORY / pdf_file
            progress_info = f"  📄 {pdf_idx + 1}/{len(batch_files)}: {pdf_file[:40]}..."
            
            pdf_data = process_pdf(pdf_path, model=MODEL, progress_info=progress_info, max_workers=MAX_WORKERS)
            
            if pdf_data:
                # Use PDF filename (without extension) as key
                pdf_key = Path(pdf_file).stem
                batch_dataset[pdf_key] = pdf_data
                
                # Update stats
                batch_stats["chunks"] += len(pdf_data)
                batch_stats["qa_pairs"] += sum(
                    len(chunk_data.get("generated", [])) 
                    for chunk_data in pdf_data.values()
                )
        
        # Save batch to JSON file
        output_filename = f"science_training_batch_{batch_num + 1:03d}.json"
        output_path = OUTPUT_DIRECTORY / output_filename
        
        print(f"  💾 Saving to {output_filename}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(batch_dataset, f, indent=2, ensure_ascii=False)
        
        # Update overall stats
        overall_stats["total_pdfs"] += len(batch_files)
        overall_stats["total_chunks"] += batch_stats["chunks"]
        overall_stats["total_qa_pairs"] += batch_stats["qa_pairs"]
        
        # Estimate cost for this batch
        estimated_input_tokens = batch_stats["chunks"] * 1500
        estimated_output_tokens = batch_stats["qa_pairs"] * 100
        batch_cost = (estimated_input_tokens / 1_000_000) * 0.15 + (estimated_output_tokens / 1_000_000) * 0.60
        overall_stats["total_cost"] += batch_cost
        
        batch_time = time.time() - batch_start_time
        print(f"  ✅ Batch {batch_num + 1} complete: {batch_stats['chunks']} chunks, {batch_stats['qa_pairs']} Q&A pairs (${batch_cost:.2f}) - {batch_time:.1f}s")
        print()
    
    # Final summary
    total_time = time.time() - start_time
    print(f"{Fore.GREEN}🎉 ALL BATCHES COMPLETE!{Fore.RESET}")
    print(f"{Fore.GREEN}📄 Total PDFs processed: {overall_stats['total_pdfs']}{Fore.RESET}")
    print(f"{Fore.GREEN}📦 Total JSON files created: {total_batches}{Fore.RESET}")
    print(f"{Fore.GREEN}🧩 Total chunks: {overall_stats['total_chunks']}{Fore.RESET}")
    print(f"{Fore.GREEN}❓ Total Q&A pairs: {overall_stats['total_qa_pairs']}{Fore.RESET}")
    print(f"{Fore.GREEN}💾 Output directory: {OUTPUT_DIRECTORY}{Fore.RESET}")
    print(f"{Fore.YELLOW}💰 Estimated total cost: ${overall_stats['total_cost']:.2f}{Fore.RESET}")
    print(f"{Fore.MAGENTA}⏱️ Total processing time: {total_time/60:.1f} minutes{Fore.RESET}")
    print(f"{Fore.MAGENTA}⚡ Average speed: {overall_stats['total_chunks']/(total_time/60):.1f} chunks/minute{Fore.RESET}")

if __name__ == "__main__": 
    main() 