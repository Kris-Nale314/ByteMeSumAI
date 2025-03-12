#!/usr/bin/env python
"""
Basic example of how to use SumWiseAI to process a document.

This example shows the minimal code needed to:
1. Load a document
2. Generate a summary
3. Output the results
"""

import os
import sys
import argparse

# Add the repo root to the Python path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import SumWiseAI

def main(document_path: str, output_dir: str, summarization_method: str = "basic") -> None:
    """
    Process a document using SumWiseAI.
    
    Args:
        document_path: Path to the document file
        output_dir: Directory to save the output
        summarization_method: Method to use for summarization
    """
    print(f"Processing document: {document_path}")
    print(f"Using summarization method: {summarization_method}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load the document
    document = Document.from_file(document_path)
    print(f"Loaded document: {len(document.content):,} characters")
    
    # Step 2: Generate a summary using the specified method
    summary_result = summarize(document, strategy=summarization_method)
    
    # Step 3: Output the summary
    print("\n" + "="*50)
    print(f"SUMMARY ({summarization_method}):")
    print("="*50)
    print(summary_result.summary)
    print("="*50)
    
    # Step 4: Save the summary to a file
    output_filename = os.path.join(output_dir, f"{os.path.basename(document_path)}.{summarization_method}_summary.txt")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Summary of: {document_path}\n")
        f.write(f"Method: {summarization_method}\n")
        f.write("="*50 + "\n")
        f.write(summary_result.summary + "\n")
        f.write("="*50 + "\n\n")
        f.write(f"Processing time: {summary_result.processing_time:.2f} seconds\n")
        f.write(f"Word count (original): {document.word_count}\n")
        f.write(f"Word count (summary): {summary_result.word_count}\n")
    
    print(f"\nSummary saved to: {output_filename}")
    
    # Step 5: If document is long, demonstrate chunking
    if len(document.content) > 5000:
        print("\nDocument is long enough for chunking. Demonstrating boundary-aware chunking:")
        chunking_result = chunk(document, strategy="boundary_aware")
        
        print(f"Document divided into {chunking_result.chunk_count} chunks")
        
        # Save chunking info
        chunks_filename = os.path.join(output_dir, f"{os.path.basename(document_path)}.chunks.txt")
        with open(chunks_filename, "w", encoding="utf-8") as f:
            f.write(f"Chunking results for: {document_path}\n")
            f.write(f"Strategy: boundary_aware\n")
            f.write("="*50 + "\n\n")
            
            for i, chunk in enumerate(chunking_result.chunks):
                f.write(f"CHUNK {i+1}:\n")
                f.write(f"Length: {chunk.length} characters\n")
                f.write(f"Position: {chunk.start_idx}-{chunk.end_idx}\n")
                boundary_type = chunk.metadata.get("boundary_type", "N/A")
                f.write(f"Boundary type: {boundary_type}\n")
                f.write("-"*30 + "\n")
                # Write the first 100 characters of each chunk
                f.write(chunk.text[:100] + "...\n\n")
        
        print(f"Chunking information saved to: {chunks_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a document with SumWiseAI")
    
    parser.add_argument("--document", "-d", type=str, required=True,
                       help="Path to the document file")
    parser.add_argument("--output", "-o", type=str, default="output",
                       help="Directory to save the output (default: 'output')")
    parser.add_argument("--method", "-m", type=str, default="basic",
                       choices=["basic", "extractive", "entity_focused", "temporal"],
                       help="Summarization method to use (default: 'basic')")
    
    args = parser.parse_args()
    
    main(args.document, args.output, args.method)