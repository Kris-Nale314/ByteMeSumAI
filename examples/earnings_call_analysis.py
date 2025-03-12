#!/usr/bin/env python
"""
Example script that demonstrates ByteMeSumAI's capabilities by analyzing
an earnings call transcript from Dell.

This script:
1. Loads an earnings call transcript
2. Processes it using boundary-aware chunking
3. Generates summaries using our simplified implementation
4. Creates basic output of the results
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Optional

# Import ByteMeSumAI modules
from bytemesumai.models.document import Document
from bytemesumai.processing.document import DocumentProcessor
from bytemesumai.chunking.processor import ChunkingProcessor
from bytemesumai.summarization.processor import SummarizationProcessor


def main(transcript_path: str, output_dir: str) -> None:
    """
    Process an earnings call transcript and generate a report.
    
    Args:
        transcript_path: Path to the transcript file
        output_dir: Directory to save the output report
    """
    print(f"Loading transcript from: {transcript_path}")
    
    # Step 1: Load the document
    try:
        document = Document(content=open(transcript_path, "r", encoding="utf-8").read(), filepath=transcript_path)
        print(f"Loaded document: {len(document.content):,} characters, approx. {len(document.content.split()):,} words")
    except Exception as e:
        print(f"Error loading document: {e}")
        return
    
    # Step 2: Create processor
    processor = DocumentProcessor()
    
    # Step 3: Process the document
    print("Processing document...")
    start_time = time.time()
    
    results = processor.process_document(
        document=document,
        chunking_strategy="fixed_size",  # Using simpler strategy for now
        summarization_strategies=["basic"]
    )
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Step 4: Generate output
    print("\n" + "="*50)
    print("DOCUMENT SUMMARY")
    print("="*50)
    
    if "summarization_result" in results:
        if isinstance(results["summarization_result"], dict) and "basic" in results["summarization_result"]:
            # For shorter documents processed directly
            print(results["summarization_result"]["basic"]["summary"])
        elif "basic_summary" in results["summarization_result"]:
            # For longer documents processed with chunking
            print(results["summarization_result"]["basic_summary"])
    
    # Step 5: Save output to a file
    output_path = os.path.join(output_dir, f"{os.path.basename(transcript_path).split('.')[0]}_analysis.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write("EARNINGS CALL ANALYSIS\n")
        f.write("="*50 + "\n\n")
        
        # Extract company name from filename
        company_name = os.path.basename(transcript_path).split('_')[0].capitalize()
        f.write(f"Company: {company_name}\n\n")
        
        # Add summary
        f.write("SUMMARY:\n")
        if "summarization_result" in results:
            if isinstance(results["summarization_result"], dict) and "basic" in results["summarization_result"]:
                f.write(results["summarization_result"]["basic"]["summary"])
            elif "basic_summary" in results["summarization_result"]:
                f.write(results["summarization_result"]["basic_summary"])
        
        # Add document and processing info
        f.write("\n\n")
        f.write("DOCUMENT INFORMATION:\n")
        f.write(f"- Document length: {len(document.content):,} characters\n")
        f.write(f"- Word count: {len(document.content.split()):,} words\n")
        f.write(f"- Processing time: {processing_time:.2f} seconds\n")
        
        # Add chunking info if available
        if "chunking_result" in results:
            f.write("\nCHUNKING INFORMATION:\n")
            f.write(f"- Strategy: {results['chunking_result']['strategy']}\n")
            f.write(f"- Chunk count: {results['chunking_result']['chunk_count']}\n")
    
    print(f"\nAnalysis saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an earnings call transcript with ByteMeSumAI")
    
    parser.add_argument("--transcript", "-t", type=str, default="examples/data/DELL_EarningsCall.txt",
                       help="Path to the earnings call transcript file")
    parser.add_argument("--output", "-o", type=str, default="output",
                       help="Directory to save the output report (default: 'output')")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    main(args.transcript, args.output)