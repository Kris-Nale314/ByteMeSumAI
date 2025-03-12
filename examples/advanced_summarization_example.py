#!/usr/bin/env python
"""
Advanced summarization demo for ByteMeSumAI.

This script demonstrates the powerful features of ByteMeSumAI's enhanced
summarization processor, including:

1. Multi-strategy summarization (basic, extractive, entity-focused, temporal)
2. Comparative summarization for multiple documents
3. Contrastive summarization for two documents
4. Summary evaluation and quality assessment
5. Visual comparison of different summarization methods

Usage:
    python advanced_summarization_demo.py --document-dir your_documents_folder --output output_folder
"""

import os
import sys
import time
import argparse
import json
import glob
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import ByteMeSumAI modules
from bytemesumai.models.document import Document
from bytemesumai.summarization.processor import SummarizationProcessor, SummaryResult
from bytemesumai.chunking.processor import ChunkingProcessor


def create_radar_chart(metrics: Dict[str, Dict[str, float]], output_path: str, title: str = "Summary Quality Comparison"):
    """
    Create a radar chart comparing metrics across different methods.
    
    Args:
        metrics: Dictionary of metrics by method
        output_path: Path to save the chart
        title: Chart title
    """
    try:
        methods = list(metrics.keys())
        metrics_names = ['Completeness', 'Conciseness', 'Accuracy', 'Coherence']
        
        # Number of variables
        N = len(metrics_names)
        
        # Create angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Add each method's data to the chart
        for i, method in enumerate(methods):
            values = []
            for metric in metrics_names:
                metric_key = f"{metric.lower()}_score"
                metric_value = metrics[method].get(metric_key, 0)
                # Convert to float if it's a string
                if isinstance(metric_value, str):
                    try:
                        metric_value = float(metric_value)
                    except ValueError:
                        metric_value = 0
                values.append(metric_value)
            
            values += values[:1]  # Close the loop
            
            # Plot the values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=method.replace('_', ' ').title())
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels and ticks
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'])
        ax.set_ylim(0, 10)
        
        # Add title and legend
        plt.title(title, size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Radar chart saved to: {output_path}")
    except Exception as e:
        print(f"Error creating radar chart: {e}")
        print("This is non-critical; continuing with other tasks.")


def demo_multi_strategy_summarization(document: Document, output_dir: str, evaluate: bool = False):
    """
    Demonstrate multiple summarization strategies on a single document.
    
    Args:
        document: Document to summarize
        output_dir: Directory to save output
        evaluate: Whether to evaluate summaries (can be slow)
    """
    print(f"\n{'='*80}\nDEMO 1: MULTI-STRATEGY SUMMARIZATION\n{'='*80}")
    
    summarizer = SummarizationProcessor()
    
    # Define methods to demonstrate
    methods = [
        ("basic", "concise", "Basic - Concise"),
        ("basic", "detailed", "Basic - Detailed"),
        ("basic", "bullet", "Basic - Bullet Points"),
        ("extractive", None, "Extractive"),
        ("entity_focused", None, "Entity-Focused"),
        ("temporal", None, "Temporal")
    ]
    
    # Generate summaries and evaluate each one
    results = {}
    evaluations = {}
    
    for method, style, label in methods:
        print(f"\nGenerating {label} summary...")
        
        # Generate summary based on method
        if method == "basic":
            result = summarizer.basic_summary(document.content, style=style)
        elif method == "extractive":
            result = summarizer.extractive_summary(document.content)
        elif method == "entity_focused":
            result = summarizer.entity_focused_summary(document.content)
        elif method == "temporal":
            result = summarizer.temporal_summary(document.content)
        
        # Store result
        results[label] = result
        
        # Print summary stats
        print(f"  - Word count: {result.word_count}")
        print(f"  - Processing time: {result.processing_time:.2f} seconds")
        
        # Evaluate summary if requested
        if evaluate:
            print(f"Evaluating {label} summary...")
            try:
                evaluation = summarizer.evaluate_summary(document.content, result.summary)
                evaluations[label] = evaluation
                if "overall_score" in evaluation:
                    print(f"  - Overall quality score: {evaluation['overall_score']:.1f}/10")
            except Exception as e:
                print(f"  - Error evaluating summary: {e}")
    
    
    # Create comparison report
    report_path = os.path.join(output_dir, "multi_strategy_comparison.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Multi-Strategy Summarization Comparison\n\n")
        f.write(f"*Generated with ByteMeSumAI on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        
        f.write(f"## Document Information\n\n")
        f.write(f"- **Document**: {document.filename or 'Unnamed document'}\n")
        f.write(f"- **Length**: {len(document.content):,} characters\n")
        f.write(f"- **Word Count**: {document.word_count:,} words\n\n")
        
        # Add quality metrics if evaluations were done
        if evaluations:
            f.write("## Summary Quality Metrics\n\n")
            f.write("| Method | Completeness | Conciseness | Accuracy | Coherence | Overall |\n")
            f.write("|--------|-------------|-------------|----------|-----------|--------|\n")
            
            for label in results.keys():
                if label in evaluations:
                    eval_result = evaluations[label]
                    f.write(f"| {label} | {eval_result.get('completeness_score', 'N/A')} | ")
                    f.write(f"{eval_result.get('conciseness_score', 'N/A')} | ")
                    f.write(f"{eval_result.get('accuracy_score', 'N/A')} | ")
                    f.write(f"{eval_result.get('coherence_score', 'N/A')} | ")
                    f.write(f"{eval_result.get('overall_score', 'N/A')} |\n")
            
            f.write("\n")
        
        # Add summaries
        for label, result in results.items():
            f.write(f"\n## {label} Summary\n\n")
            f.write(f"{result.summary}\n\n")
            f.write(f"*Word count: {result.word_count}, Processing time: {result.processing_time:.2f}s*\n\n")
            
            # Add evaluation details if available
            if label in evaluations:
                eval_result = evaluations[label]
                
                f.write("### Evaluation\n\n")
                
                if "strengths" in eval_result and eval_result["strengths"]:
                    f.write("**Strengths:**\n")
                    for strength in eval_result["strengths"]:
                        f.write(f"- {strength}\n")
                    f.write("\n")
                
                if "improvement_suggestions" in eval_result and eval_result["improvement_suggestions"]:
                    f.write("**Improvement Suggestions:**\n")
                    for suggestion in eval_result["improvement_suggestions"]:
                        f.write(f"- {suggestion}\n")
                    f.write("\n")
    
    print(f"\nComparison report saved to: {report_path}")
    
    # Create radar chart for visual comparison if evaluations are available
    if evaluations:
        chart_path = os.path.join(output_dir, "summary_quality_comparison.png")
        create_radar_chart(evaluations, chart_path)
    
    return results, evaluations


def demo_document_segmentation_and_summary(document: Document, output_dir: str):
    """
    Demonstrate document segmentation and summarization.
    
    Args:
        document: Document to process
        output_dir: Directory to save output
    """
    print(f"\n{'='*80}\nDEMO 2: DOCUMENT SEGMENTATION AND SUMMARIZATION\n{'='*80}")
    
    # Create processors
    chunker = ChunkingProcessor()
    summarizer = SummarizationProcessor()
    
    # Process with boundary-aware chunking
    print("Performing boundary-aware chunking...")
    chunking_result = chunker.chunk_document(document.content, strategy="boundary_aware")
    chunks = chunking_result.chunks  # Access chunks as property instead of dictionary access
    
    print(f"Document divided into {len(chunks)} segments based on natural boundaries")
    
    # Summarize each segment
    print("Summarizing each segment...")
    segment_summaries = []
    
    for i, chunk in enumerate(chunks):
        print(f"  Processing segment {i+1}/{len(chunks)}...")
        
        # Get chunk text
        chunk_text = chunk.text
        
        # Generate summary
        summary = summarizer.basic_summary(chunk_text, style="concise")
        
        # Store result
        segment_summaries.append({
            "segment_index": i,
            "segment_text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
            "segment_length": len(chunk_text),
            "segment_word_count": len(chunk_text.split()),
            "summary": summary.summary,
            "summary_word_count": summary.word_count
        })
    
    # Generate hierarchical summary
    print("Generating hierarchical summary...")
    
    # First, combine segment summaries
    combined_segment_summaries = "\n\n".join([f"SEGMENT {i+1}:\n{s['summary']}" 
                                          for i, s in enumerate(segment_summaries)])
    
    # Then, summarize the combined summaries
    hierarchical_summary = summarizer.basic_summary(
        combined_segment_summaries,
        style="detailed"
    )
    
    # Create report
    report_path = os.path.join(output_dir, "hierarchical_summary.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Hierarchical Document Summary\n\n")
        f.write(f"*Generated with ByteMeSumAI on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        
        f.write(f"## Document Information\n\n")
        f.write(f"- **Document**: {document.filename or 'Unnamed document'}\n")
        f.write(f"- **Length**: {len(document.content):,} characters\n")
        f.write(f"- **Word Count**: {document.word_count:,} words\n")
        f.write(f"- **Segments**: {len(chunks)}\n\n")
        
        # Add chunking metrics if available
        if chunking_result.metrics:  # Access metrics as property
            f.write(f"- **Boundary Preservation**: {chunking_result.metrics.get('boundary_preservation_score', 'N/A')}\n")
            f.write(f"- **Sentence Integrity**: {chunking_result.metrics.get('sentence_integrity_score', 'N/A')}\n\n")
        
        f.write("## Overall Summary\n\n")
        f.write(f"{hierarchical_summary.summary}\n\n")
        
        f.write("## Segment Summaries\n\n")
        
        for i, segment in enumerate(segment_summaries):
            f.write(f"### Segment {i+1}\n\n")
            f.write(f"{segment['summary']}\n\n")
            f.write(f"*Segment length: {segment['segment_length']} chars, {segment['segment_word_count']} words*\n\n")
    
    print(f"\nHierarchical summary report saved to: {report_path}")
    
    return {
        "segment_summaries": segment_summaries,
        "hierarchical_summary": hierarchical_summary.summary
    }


def demo_multi_document_summarization(documents: List[Document], output_dir: str):
    """
    Demonstrate multi-document summarization.
    
    Args:
        documents: List of documents to summarize
        output_dir: Directory to save output
    """
    print(f"\n{'='*80}\nDEMO 3: MULTI-DOCUMENT SUMMARIZATION\n{'='*80}")
    
    if len(documents) < 2:
        print("Need at least 2 documents for multi-document summarization")
        return None
    
    summarizer = SummarizationProcessor()
    
    # Prepare documents
    doc_objects = []
    for doc in documents:
        doc_objects.append({
            "content": doc.content,
            "title": doc.filename or "Unnamed document"
        })
    
    # Generate summaries with different focuses
    focus_types = ["comparative", "integrative", "contrastive"]
    multi_doc_summaries = {}
    
    for focus in focus_types:
        print(f"\nGenerating {focus} multi-document summary...")
        
        summary = summarizer.multi_document_summary(
            doc_objects,
            focus=focus
        )
        
        multi_doc_summaries[focus] = summary
    
    # If we have exactly 2 documents, also do a direct contrastive summary
    if len(documents) == 2:
        print("\nGenerating direct contrastive summary between two documents...")
        
        contrastive_summary = summarizer.contrastive_summary(
            documents[0].content,
            documents[1].content
        )
        
        multi_doc_summaries["direct_contrastive"] = contrastive_summary
    
    # Create report
    report_path = os.path.join(output_dir, "multi_document_summary.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Multi-Document Summary\n\n")
        f.write(f"*Generated with ByteMeSumAI on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        
        f.write(f"## Documents Analyzed\n\n")
        for i, doc in enumerate(documents):
            f.write(f"- **Document {i+1}**: {doc.filename or 'Unnamed document'}\n")
            f.write(f"  - Length: {len(doc.content):,} characters\n")
            f.write(f"  - Word Count: {doc.word_count:,} words\n\n")
        
        # Add summaries
        for focus, summary in multi_doc_summaries.items():
            title = focus.replace('_', ' ').title()
            f.write(f"\n## {title} Summary\n\n")
            f.write(f"{summary.summary}\n\n")
            f.write(f"*Word count: {summary.word_count}, Processing time: {summary.processing_time:.2f}s*\n\n")
    
    print(f"\nMulti-document summary report saved to: {report_path}")
    
    return multi_doc_summaries


def main(document_dir: str, output_dir: str, evaluate: bool = False):
    """
    Run all demonstration examples.
    
    Args:
        document_dir: Directory containing document files
        output_dir: Directory to save output
        evaluate: Whether to evaluate summaries (can be slow)
    """
    # Find document files
    document_files = glob.glob(os.path.join(document_dir, "*.txt"))
    
    if not document_files:
        print(f"No document files found in {document_dir}")
        return
    
    print(f"Found {len(document_files)} document files")
    
    # Load documents
    documents = []
    for file_path in document_files:
        try:
            document = Document.from_file(file_path)
            documents.append(document)
            print(f"Loaded document: {document.filename} ({document.word_count:,} words)")
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
    
    if not documents:
        print("No valid documents loaded")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run demos
    demo1_results = None
    demo2_results = None
    demo3_results = None
    
    # Demo 1: Multi-strategy summarization (using first document)
    demo1_results = demo_multi_strategy_summarization(documents[0], output_dir, evaluate)
    
    # Demo 2: Document segmentation and summarization (using first document)
    demo2_results = demo_document_segmentation_and_summary(documents[0], output_dir)
    
    # Demo 3: Multi-document summarization (if multiple documents available)
    if len(documents) >= 2:
        demo3_results = demo_multi_document_summarization(documents, output_dir)
    else:
        print("\nSkipping multi-document summarization demo (need at least 2 documents)")
    
    # Create index report
    index_path = os.path.join(output_dir, "index.md")
    
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(f"# ByteMeSumAI Advanced Summarization Demo\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        
        f.write("## Demos Included\n\n")
        f.write("1. [Multi-Strategy Summarization](multi_strategy_comparison.md)\n")
        f.write("2. [Hierarchical Document Summarization](hierarchical_summary.md)\n")
        
        if demo3_results:
            f.write("3. [Multi-Document Summarization](multi_document_summary.md)\n")
        
        f.write("\n## Documents Processed\n\n")
        for doc in documents:
            f.write(f"- {doc.filename or 'Unnamed document'} ({doc.word_count:,} words)\n")
    
    print(f"\nDemo index saved to: {index_path}")
    print(f"\nAll demos completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Summarization Demo for ByteMeSumAI")
    
    parser.add_argument("--document-dir", "-d", type=str, default="examples/data",
                       help="Directory containing document files")
    parser.add_argument("--output", "-o", type=str, default="output",
                       help="Directory to save the output reports (default: 'output')")
    parser.add_argument("--evaluate", "-e", action="store_true",
                       help="Evaluate summaries (can be slow)")
    
    args = parser.parse_args()
    
    main(args.document_dir, args.output, args.evaluate)