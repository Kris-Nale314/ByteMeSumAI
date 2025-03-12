#!/usr/bin/env python
"""
Simplified example script that demonstrates SumWiseAI's capabilities
by analyzing an earnings call transcript from Dell.

This script focuses on core functionality to get started quickly.
"""

import os
import sys
import time
import argparse

# Import SumWiseAI
from sumwiseai.models.document import Document
from sumwiseai.processing.document import DocumentProcessor

def main(transcript_path, output_dir):
    """Process an earnings call transcript and print results."""
    print(f"Loading transcript from: {transcript_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load the document
    try:
        document = Document(content=open(transcript_path, "r").read(), filepath=transcript_path)
        print(f"Loaded document: {len(document.content):,} characters")
    except Exception as e:
        print(f"Error loading document: {e}")
        return
    
    # Step 2: Process the document
    processor = DocumentProcessor()
    
    print("Processing document...")
    start_time = time.time()
    
    results = processor.process_document(
        document=document,
        chunking_strategy="fixed_size",  # Using simpler strategy for now
        summarization_strategies=["basic"]
    )
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Step 3: Output results
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
    
    # Step 4: Save results to a file
    output_path = os.path.join(output_dir, f"{os.path.basename(transcript_path)}_summary.txt")
    
    with open(output_path, "w") as f:
        f.write("="*50 + "\n")
        f.write("DOCUMENT SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        if "summarization_result" in results:
            if isinstance(results["summarization_result"], dict) and "basic" in results["summarization_result"]:
                f.write(results["summarization_result"]["basic"]["summary"])
            elif "basic_summary" in results["summarization_result"]:
                f.write(results["summarization_result"]["basic_summary"])
        
        f.write("\n\n")
        f.write(f"Processing time: {processing_time:.2f} seconds\n")
        f.write(f"Document length: {len(document.content):,} characters\n")
    
    print(f"\nSummary saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an earnings call transcript with SumWiseAI")
    
    parser.add_argument("--transcript", "-t", type=str, default="examples/data/Dell_Q4_2024_EarningsCall.txt",
                      help="Path to the earnings call transcript file")
    parser.add_argument("--output", "-o", type=str, default="output",
                      help="Directory to save the output (default: 'output')")
    
    args = parser.parse_args()
    
    main(args.transcript, args.output)