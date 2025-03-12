# SumWiseAI Examples

This directory contains example scripts that demonstrate how to use the SumWiseAI package.

## Prerequisites

Make sure you have SumWiseAI installed:

```bash
pip install -e ..  # Install from local directory in development mode
```

You'll also need to set up your API keys for the LLM providers you want to use. The examples use environment variables for API keys:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic (optional)
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Examples

### Earnings Call Analysis

Process an earnings call transcript with boundary-aware chunking and multi-strategy summarization:

```bash
python earnings_call_analysis.py --transcript data/Dell_Q4_2024_EarningsCall.txt --output output
```

This will:
1. Load the transcript file
2. Process it using boundary-aware chunking
3. Generate summaries using multiple strategies (basic, extractive, entity-focused, temporal)
4. Create a markdown report with the results

### Basic Document Processing

Simple example of using SumWiseAI for basic document processing:

```bash
python basic_document_processing.py --document your_document.txt --output output
```

## Data Directory

The `data` directory contains sample documents for testing:

- `Dell_Q4_2024_EarningsCall.txt`: Sample earnings call transcript
- Add more test files here...

## Output

Results will be saved to the specified output directory (defaults to `output/`).