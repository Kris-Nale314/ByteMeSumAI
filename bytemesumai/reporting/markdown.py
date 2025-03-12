"""
Markdown reporter for ByteMeSumAI.

This module provides functionality for generating markdown reports from
document processing results.
"""

import time
import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

class MarkdownReporter:
    """
    Generate markdown reports from document processing results.
    """
    
    def generate_report(self, results: Dict[str, Any], title: str = "Document Analysis Report") -> str:
        """
        Generate a markdown report from document processing results.
        
        Args:
            results: Results from DocumentProcessor.process_document()
            title: Report title
            
        Returns:
            Markdown content
        """
        md_lines = []
        
        # Add title and introduction
        md_lines.append(f"# {title}\n")
        
        # Add generation timestamp
        md_lines.append(f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # Document information
        document_info = results.get("document_info", {})
        md_lines.append("## Document Information\n")
        
        if document_info:
            md_lines.append(f"- **Length**: {document_info.get('length', 'N/A'):,} characters\n")
            md_lines.append(f"- **Word Count**: Approximately {document_info.get('word_count', 'N/A'):,} words\n")
            
            # Add filename if available
            if "filename" in document_info and document_info["filename"]:
                md_lines.append(f"- **Source**: {document_info['filename']}\n")
        else:
            md_lines.append("*Document information not available.*\n")
        
        # Processing information
        processing_info = results.get("processing_info", {})
        if processing_info:
            md_lines.append("\n## Processing Information\n")
            md_lines.append(f"- **Chunking Strategy**: {processing_info.get('chunking_strategy', 'N/A')}\n")
            md_lines.append(f"- **Summarization Strategies**: {', '.join(processing_info.get('summarization_strategies', ['N/A']))}\n")
            md_lines.append(f"- **Processing Time**: {processing_info.get('total_processing_time', 0):.2f} seconds\n")
            
            # Add long document handling information
            if processing_info.get("is_long_document", False):
                md_lines.append("- **Long Document**: Yes (processed with chunking)\n")
            else:
                md_lines.append("- **Long Document**: No (processed without chunking)\n")
        
        # Chunking results
        chunking_result = results.get("chunking_result")
        if chunking_result:
            md_lines.append("\n## Document Chunking\n")
            md_lines.append(f"- **Strategy**: {chunking_result.get('strategy', 'N/A').replace('_', ' ').title()}\n")
            md_lines.append(f"- **Chunks Created**: {chunking_result.get('chunk_count', 0)}\n")
            
            # Add boundary information if available
            if "boundaries" in chunking_result and chunking_result["boundaries"]:
                boundaries = chunking_result["boundaries"]
                md_lines.append(f"- **Boundaries Detected**: {len(boundaries)}\n")
                
            # Add chunking metrics if available
            metrics = chunking_result.get("metrics", {})
            if metrics:
                md_lines.append("\n### Chunking Metrics\n")
                md_lines.append(f"- **Average Chunk Size**: {metrics.get('avg_chunk_size', 0):.1f} characters\n")
                md_lines.append(f"- **Min Chunk Size**: {metrics.get('min_chunk_size', 0):,} characters\n")
                md_lines.append(f"- **Max Chunk Size**: {metrics.get('max_chunk_size', 0):,} characters\n")
                
                # Add boundary preservation score if available
                if "boundary_preservation_score" in metrics:
                    score = metrics["boundary_preservation_score"]
                    md_lines.append(f"- **Boundary Preservation Score**: {score:.2f} ")
                    if score > 0.9:
                        md_lines.append("(Excellent)\n")
                    elif score > 0.7:
                        md_lines.append("(Good)\n")
                    else:
                        md_lines.append("(Moderate)\n")
                
                # Add sentence integrity score if available
                if "sentence_integrity_score" in metrics:
                    score = metrics["sentence_integrity_score"]
                    md_lines.append(f"- **Sentence Integrity Score**: {score:.2f} ")
                    if score > 0.9:
                        md_lines.append("(Excellent)\n")
                    elif score > 0.7:
                        md_lines.append("(Good)\n")
                    else:
                        md_lines.append("(Moderate)\n")
            
            # Add information about chunks in a collapsible section
            if "chunks" in chunking_result and chunking_result["chunks"]:
                chunks = chunking_result["chunks"]
                md_lines.append("\n<details>")
                md_lines.append("<summary><b>Chunk Details</b> (Click to expand)</summary>\n")
                
                md_lines.append("\n| Chunk | Length | Start | End | Boundary Type |\n")
                md_lines.append("|-------|--------|-------|-----|---------------|\n")
                
                for i, chunk in enumerate(chunks):
                    length = chunk.get("length", 0)
                    start = chunk.get("start_idx", "N/A")
                    end = chunk.get("end_idx", "N/A")
                    
                    # Get boundary type if available
                    boundary_type = "N/A"
                    if "metadata" in chunk and chunk["metadata"]:
                        boundary_type = chunk["metadata"].get("boundary_type", "N/A")
                    
                    md_lines.append(f"| {i+1} | {length:,} chars | {start:,} | {end:,} | {boundary_type} |\n")
                
                md_lines.append("</details>\n")
        
        # Summarization results
        summarization_result = results.get("summarization_result", {})
        if summarization_result:
            md_lines.append("\n## Summarization Results\n")
            
            # Get results for each strategy
            strategy_results = summarization_result.get("results", {})
            
            # Add basic summary first if available
            if "basic" in strategy_results:
                basic_result = strategy_results["basic"]
                md_lines.append("### Executive Summary\n")
                md_lines.append(basic_result.get("summary", "*No summary available.*") + "\n")
                
                # Add style information if available
                if "style" in basic_result:
                    md_lines.append(f"*Summary style: {basic_result['style']}*\n")
                
                # Add hierarchical information if available
                if basic_result.get("method", "") == "hierarchical":
                    md_lines.append("*Generated using hierarchical boundary-aware summarization*\n")
                    
                    # Add chunk summaries in a collapsible section
                    if "chunk_summaries" in basic_result and basic_result["chunk_summaries"]:
                        md_lines.append("\n<details>")
                        md_lines.append("<summary><b>Section-by-Section Summaries</b> (Click to expand)</summary>\n")
                        
                        for i, summary in enumerate(basic_result["chunk_summaries"]):
                            md_lines.append(f"\n#### Section {i+1} Summary\n")
                            md_lines.append(f"{summary}\n")
                        
                        md_lines.append("</details>\n")
            
            # Add extractive summary
            if "extractive" in strategy_results:
                extractive_result = strategy_results["extractive"]
                md_lines.append("\n### Key Statements\n")
                
                if extractive_result.get("method", "").startswith("hierarchical"):
                    md_lines.append("The following statements represent the most important information from each section of the document:\n")
                else:
                    md_lines.append("The following statements represent the most important information in the document:\n")
                    
                md_lines.append(extractive_result.get("summary", "*No extractive summary available.*") + "\n")
                
                # Add compression information if available
                if "compression_ratio" in extractive_result:
                    ratio = extractive_result["compression_ratio"]
                    md_lines.append(f"*Compression ratio: {ratio:.2f} (lower means more concise)*\n")
            
            # Add entity-focused summary
            if "entity_focused" in strategy_results:
                entity_result = strategy_results["entity_focused"]
                md_lines.append("\n### Key Entities and Their Context\n")
                md_lines.append(entity_result.get("summary", "*No entity-focused summary available.*") + "\n")
                
                # List detected entities
                entities = entity_result.get("entities", [])
                if entities:
                    md_lines.append("\n#### Detected Key Entities\n")
                    for entity in entities:
                        md_lines.append(f"- {entity}\n")
                
                # Add hierarchical information if available
                if entity_result.get("method", "").startswith("hierarchical"):
                    md_lines.append("\n*Generated using boundary-aware entity extraction and analysis*\n")
            
            # Add temporal summary
            if "temporal" in strategy_results:
                temporal_result = strategy_results["temporal"]
                md_lines.append("\n### Chronological Analysis\n")
                md_lines.append(temporal_result.get("summary", "*No temporal summary available.*") + "\n")
                
                # List detected time periods
                time_periods = temporal_result.get("time_periods", [])
                if time_periods:
                    md_lines.append("\n#### Detected Time Periods\n")
                    for period in time_periods:
                        md_lines.append(f"- {period}\n")
                
                # Add ordering information
                if "chronological_order" in temporal_result:
                    order = "chronological" if temporal_result["chronological_order"] else "reverse chronological"
                    md_lines.append(f"\n*Presented in {order} order*\n")
        
        # Add evaluation results if available
        evaluations = summarization_result.get("evaluations", {})
        if evaluations:
            md_lines.append("\n## Summary Evaluation\n")
            
            md_lines.append("| Metric | Score | Interpretation |\n")
            md_lines.append("|--------|-------|---------------|\n")
            
            # Use basic summary evaluation if available
            eval_data = evaluations.get("basic", next(iter(evaluations.values())) if evaluations else {})
            
            if eval_data:
                metrics = [
                    ("Completeness", "completeness_score"),
                    ("Conciseness", "conciseness_score"),
                    ("Accuracy", "accuracy_score"),
                    ("Coherence", "coherence_score"),
                    ("Overall", "overall_score")
                ]
                
                for label, key in metrics:
                    if key in eval_data:
                        score = eval_data[key]
                        interpretation = "Excellent" if score >= 8 else "Good" if score >= 6 else "Average" if score >= 4 else "Poor"
                        md_lines.append(f"| {label} | {score:.1f}/10 | {interpretation} |\n")
                
                # Add improvement suggestions if available
                if "improvement_suggestions" in eval_data and eval_data["improvement_suggestions"]:
                    md_lines.append("\n### Improvement Suggestions\n")
                    for suggestion in eval_data["improvement_suggestions"]:
                        md_lines.append(f"- {suggestion}\n")
        
        # Add document complexity assessment
        md_lines.append("\n## Document Complexity Assessment\n")
        
        # Extract complexity metrics from results
        entity_count = 0
        if "entity_focused" in strategy_results:
            entity_count = len(strategy_results["entity_focused"].get("entities", []))
            
        time_period_count = 0
        if "temporal" in strategy_results:
            time_period_count = len(strategy_results["temporal"].get("time_periods", []))
            
        document_length = document_info.get("length", 0)
        
        # Entity complexity
        if entity_count > 5:
            complexity = "High"
        elif entity_count > 2:
            complexity = "Medium"
        else:
            complexity = "Low"
        md_lines.append(f"- **Entity Complexity**: {complexity} ({entity_count} key entities detected)\n")
        
        # Temporal complexity
        if time_period_count > 3:
            complexity = "High"
        elif time_period_count > 1:
            complexity = "Medium"
        else:
            complexity = "Low"
        md_lines.append(f"- **Temporal Complexity**: {complexity} ({time_period_count} time periods detected)\n")
        
        # Size complexity
        if document_length > 50000:
            size_complexity = "Very High"
        elif document_length > 15000:
            size_complexity = "High"
        elif document_length > 5000:
            size_complexity = "Medium"
        else:
            size_complexity = "Low"
        md_lines.append(f"- **Size Complexity**: {size_complexity} ({document_length:,} characters)\n")
        
        # Boundary complexity
        boundary_count = 0
        if chunking_result and "boundaries" in chunking_result:
            boundary_count = len(chunking_result["boundaries"] or [])
            
            if boundary_count > 10:
                boundary_complexity = "Very High"
            elif boundary_count > 5:
                boundary_complexity = "High"
            elif boundary_count > 2:
                boundary_complexity = "Medium"
            else:
                boundary_complexity = "Low"
            md_lines.append(f"- **Structural Complexity**: {boundary_complexity} ({boundary_count} document boundaries detected)\n")
        
        # Add RAG strategy recommendations
        md_lines.append("\n## RAG Strategy Recommendations\n")
        recommendations = []
        
        # Size-based recommendations
        if document_length > 15000:
            recommendations.append("- **Boundary-Aware Chunking**: This document benefits significantly from intelligent chunking.")
            recommendations.append("- **Hierarchical Processing**: Process the document in stages to maintain coherence between sections.")
        
        if entity_count > 3:
            recommendations.append("- **Entity-Aware Retrieval**: Use entity tracking to properly disambiguate multiple entities.")
        
        if time_period_count > 1:
            recommendations.append("- **Temporal Tracking**: Implement chronological awareness to maintain temporal consistency.")
        
        if boundary_count > 5:
            recommendations.append("- **Structure Preservation**: Respect document structure when processing to maintain context.")
            recommendations.append("- **Multi-Agent Synthesis**: Use specialized agents for different document sections or topics.")
        
        if processing_info.get("is_long_document", False):
            recommendations.append("- **Context Management**: Use techniques that preserve context across document sections.")
            recommendations.append("- **Agentic Coordination**: Employ agents to handle different aspects of the document.")
        
        if not recommendations:
            recommendations.append("- **Standard Processing**: This document is relatively simple and can be processed with standard RAG techniques.")
            recommendations.append("- **Single-Pass Analysis**: The document can be analyzed in a single pass without complex chunking or specialized handling.")
        
        md_lines.extend(recommendations)
        
        return "\n".join(md_lines)
    
    def save_report(self, markdown_content: str, output_path: str) -> str:
        """
        Save a markdown report to a file.
        
        Args:
            markdown_content: Markdown content to save
            output_path: Path to save the file
            
        Returns:
            Full path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save the report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
            
        return os.path.abspath(output_path)
    
    def generate_and_save(self, results: Dict[str, Any], output_path: str, title: str = "Document Analysis Report") -> str:
        """
        Generate and save a markdown report from document processing results.
        
        Args:
            results: Results from DocumentProcessor.process_document()
            output_path: Path to save the report
            title: Report title
            
        Returns:
            Full path to the saved file
        """
        markdown_content = self.generate_report(results, title)
        return self.save_report(markdown_content, output_path)