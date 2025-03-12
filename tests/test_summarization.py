"""
Tests for the enhanced summarization functionality in ByteMeSumAI.
"""

import pytest
from unittest.mock import MagicMock, patch
import re

from bytemesumai.summarization.processor import SummarizationProcessor, SummaryResult
from bytemesumai.llm.client import LLMClient


# Sample text for testing
SAMPLE_TEXT = """
# Introduction to Artificial Intelligence

Artificial Intelligence (AI) is a branch of computer science that aims to create systems capable of performing tasks that normally require human intelligence. These tasks include visual perception, speech recognition, decision-making, and translation between languages.

## History of AI

The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. The attendees, including John McCarthy, Marvin Minsky, Allen Newell, and Herbert Simon, became the leaders of AI research for decades.

## Types of AI

### Narrow AI
Narrow AI, also known as Weak AI, is designed to perform a narrow task such as voice recognition, image recognition, or internet searches. Most of the AI that surrounds us today is narrow AI.

### General AI
General AI, also known as Strong AI or AGI (Artificial General Intelligence), refers to a machine with general intelligence that can solve any problem that a human being can.

## Machine Learning

Machine Learning (ML) is a subset of AI that focuses on the development of algorithms that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction.

### Deep Learning

Deep Learning is a subset of ML that has networks capable of learning unsupervised from data that is unstructured or unlabeled. Also known as Deep Neural Learning or Deep Neural Network (DNN).

## Applications of AI

AI has a wide range of applications across various industries:

1. Healthcare: AI is being used for drug discovery, personalized medicine, and medical imaging analysis.
2. Finance: AI algorithms are used for fraud detection, algorithmic trading, and risk assessment.
3. Transportation: Self-driving cars and traffic management systems use AI technology.
4. Entertainment: Recommendation systems on platforms like Netflix and Spotify use AI.

## Future of AI

The future of AI holds exciting possibilities and challenges. As AI systems become more sophisticated, questions about ethics, safety, and governance become increasingly important. Ensuring that AI development benefits humanity as a whole remains a crucial consideration.

## Conclusion

Artificial Intelligence continues to evolve rapidly, transforming industries and everyday life. Understanding its capabilities, limitations, and potential impacts is essential as we move forward into an increasingly AI-driven world.
"""

SAMPLE_TEXT_2 = """
# The Impact of Machine Learning on Business

Machine Learning (ML) is revolutionizing how businesses operate and make decisions. As a subset of Artificial Intelligence, ML enables systems to learn from data and improve without explicit programming.

## Business Applications of Machine Learning

### Customer Insights
Businesses use ML to analyze customer behavior and preferences, enabling personalized marketing and improved customer experiences. Recommendation systems are a prime example, used by companies like Amazon and Netflix.

### Process Automation
ML facilitates automation of routine tasks, reducing operational costs and minimizing human error. This includes document processing, data entry, and basic customer service interactions.

### Predictive Analytics
Companies leverage ML for forecasting trends, sales projections, and risk assessment. This helps in strategic planning and proactive decision-making.

## Implementation Challenges

Despite its benefits, implementing ML in business settings faces several challenges:

1. Data Quality: ML models require high-quality, relevant data to produce accurate results.
2. Talent Gap: There's a shortage of professionals with expertise in ML and data science.
3. Integration: Incorporating ML solutions into existing infrastructure can be complex.
4. Cost: Initial investment in ML technology and expertise can be significant.

## Future Directions

The business applications of ML will continue to expand as technology advances. Edge computing, federated learning, and AutoML are emerging trends that will make ML more accessible and efficient for businesses of all sizes.

## Conclusion

Machine Learning is not just a technological innovation but a strategic business asset. Companies that successfully implement ML solutions gain competitive advantages through improved efficiency, enhanced customer experiences, and data-driven decision-making capabilities.
"""


def test_summarization_processor_initialization():
    """Test SummarizationProcessor initialization with default parameters."""
    processor = SummarizationProcessor()
    assert processor.default_model == "gpt-3.5-turbo"
    assert not processor.use_mock
    assert isinstance(processor.llm_client, LLMClient)


def test_mock_basic_summary():
    """Test basic_summary method with mock mode."""
    processor = SummarizationProcessor(use_mock=True)
    
    # Test concise style
    result = processor.basic_summary(SAMPLE_TEXT, style="concise")
    assert isinstance(result, SummaryResult)
    assert result.method == "basic"
    assert len(result.summary) > 0
    assert result.metadata.get("style") == "concise"
    
    # Test detailed style
    result = processor.basic_summary(SAMPLE_TEXT, style="detailed")
    assert "detailed" in result.summary.lower() or "additional details" in result.summary.lower()
    
    # Test bullet style
    result = processor.basic_summary(SAMPLE_TEXT, style="bullet")
    assert "•" in result.summary or "-" in result.summary
    
    # Test max_length parameter
    max_length = 50
    result = processor.basic_summary(SAMPLE_TEXT, max_length=max_length)
    assert result.word_count <= max_length


def test_mock_extractive_summary():
    """Test extractive_summary method with mock mode."""
    processor = SummarizationProcessor(use_mock=True)
    
    result = processor.extractive_summary(SAMPLE_TEXT)
    assert isinstance(result, SummaryResult)
    assert result.method == "extractive"
    assert len(result.summary) > 0
    assert "extractive" in result.summary.lower()
    
    # Check metadata
    assert "original_sentences" in result.metadata
    assert "summary_sentences" in result.metadata
    assert "compression_ratio" in result.metadata


def test_mock_entity_focused_summary():
    """Test entity_focused_summary method with mock mode."""
    processor = SummarizationProcessor(use_mock=True)
    
    # Test with provided entities
    entities = ["AI", "Machine Learning", "Deep Learning"]
    result = processor.entity_focused_summary(SAMPLE_TEXT, entities=entities)
    assert isinstance(result, SummaryResult)
    assert result.method == "entity_focused"
    assert len(result.summary) > 0
    assert "entity" in result.summary.lower() or "entities" in result.summary.lower()
    
    # Check metadata
    assert result.metadata.get("entities") == entities
    assert result.metadata.get("entity_count") == len(entities)
    
    # Test without entities (should extract them automatically)
    result = processor.entity_focused_summary(SAMPLE_TEXT)
    assert len(result.metadata.get("entities", [])) > 0


def test_mock_temporal_summary():
    """Test temporal_summary method with mock mode."""
    processor = SummarizationProcessor(use_mock=True)
    
    result = processor.temporal_summary(SAMPLE_TEXT)
    assert isinstance(result, SummaryResult)
    assert result.method == "temporal"
    assert len(result.summary) > 0
    assert "temporal" in result.summary.lower() or "chronological" in result.summary.lower()
    
    # Check metadata
    assert "time_periods" in result.metadata
    assert "chronological_order" in result.metadata
    assert result.metadata.get("chronological_order") is True


def test_mock_multi_document_summary():
    """Test multi_document_summary method with mock mode."""
    processor = SummarizationProcessor(use_mock=True)
    
    documents = [
        SAMPLE_TEXT,
        SAMPLE_TEXT_2
    ]
    
    # Test with comparative focus
    result = processor.multi_document_summary(documents, focus="comparative")
    assert isinstance(result, SummaryResult)
    assert result.method == "multi_document"
    assert len(result.summary) > 0
    assert result.metadata.get("focus") == "comparative"
    assert result.metadata.get("document_count") == 2
    
    # Test with query
    query = "How has AI evolved over time?"
    result = processor.multi_document_summary(documents, query=query)
    assert result.metadata.get("query") == query
    assert result.metadata.get("is_query_focused") is True


def test_mock_contrastive_summary():
    """Test contrastive_summary method with mock mode."""
    processor = SummarizationProcessor(use_mock=True)
    
    result = processor.contrastive_summary(SAMPLE_TEXT, SAMPLE_TEXT_2)
    assert isinstance(result, SummaryResult)
    assert result.method == "contrastive"
    assert len(result.summary) > 0
    assert "contrastive" in result.summary.lower() or "differences" in result.summary.lower()
    
    # Test with focus
    focus = "machine learning applications"
    result = processor.contrastive_summary(SAMPLE_TEXT, SAMPLE_TEXT_2, focus=focus)
    assert result.metadata.get("focus") == focus


def test_mock_evaluate_summary():
    """Test evaluate_summary method with mock mode."""
    processor = SummarizationProcessor(use_mock=True)
    
    # Generate a summary to evaluate
    summary = processor.basic_summary(SAMPLE_TEXT).summary
    
    # Evaluate the summary
    evaluation = processor.evaluate_summary(SAMPLE_TEXT, summary)
    
    # Check evaluation metrics
    assert "completeness_score" in evaluation
    assert "conciseness_score" in evaluation
    assert "accuracy_score" in evaluation
    assert "coherence_score" in evaluation
    assert "overall_score" in evaluation
    
    # Check additional fields
    assert "factual_errors" in evaluation
    assert "important_omissions" in evaluation
    assert "strengths" in evaluation
    assert "improvement_suggestions" in evaluation
    
    # Check metadata
    assert "processing_time" in evaluation
    assert "original_length" in evaluation
    assert "summary_length" in evaluation
    assert "compression_ratio" in evaluation


def test_compare_summaries():
    """Test compare_summaries method."""
    processor = SummarizationProcessor(use_mock=True)
    
    # Compare different summarization methods
    methods = ["basic", "extractive", "entity_focused"]
    comparison = processor.compare_summaries(SAMPLE_TEXT, methods=methods)
    
    # Check comparison structure
    assert "text_length" in comparison
    assert "methods" in comparison
    assert comparison["methods"] == methods
    assert "summaries" in comparison
    assert "evaluations" in comparison
    assert "comparison" in comparison
    
    # Check comparison details
    assert "length_comparison" in comparison["comparison"]
    assert "processing_time_comparison" in comparison["comparison"]
    assert "rankings" in comparison["comparison"]
    assert "best_methods" in comparison["comparison"]
    
    # Check that all methods are included
    for method in methods:
        assert method in comparison["summaries"]


@patch('bytemesumai.llm.client.LLMClient.generate_completion')
def test_real_basic_summary(mock_generate_completion):
    """Test basic_summary method with mocked LLM client."""
    # Mock the LLM response
    mock_generate_completion.return_value = "This is a test summary of the document."
    
    processor = SummarizationProcessor(use_mock=False)
    result = processor.basic_summary(SAMPLE_TEXT, style="concise")
    
    # Verify that the LLM client was called
    mock_generate_completion.assert_called_once()
    
    # Check result
    assert isinstance(result, SummaryResult)
    assert result.summary == "This is a test summary of the document."
    assert result.method == "basic"
    assert result.metadata.get("style") == "concise"


@patch('bytemesumai.llm.client.LLMClient.generate_completion')
def test_real_extractive_summary(mock_generate_completion):
    """Test extractive_summary method with mocked LLM client."""
    # Mock the LLM response
    mock_generate_completion.return_value = "This is the first important sentence. This is the second important sentence."
    
    processor = SummarizationProcessor(use_mock=False)
    result = processor.extractive_summary(SAMPLE_TEXT)
    
    # Verify that the LLM client was called
    mock_generate_completion.assert_called_once()
    
    # Check result
    assert isinstance(result, SummaryResult)
    assert result.summary == "This is the first important sentence. This is the second important sentence."
    assert result.method == "extractive"


@patch('bytemesumai.llm.client.LLMClient.generate_completion')
def test_real_entity_extraction(mock_generate_completion):
    """Test _extract_key_entities method with mocked LLM client."""
    # Mock the LLM response
    mock_generate_completion.return_value = "AI, Machine Learning, Deep Learning, John McCarthy, Dartmouth College"
    
    processor = SummarizationProcessor(use_mock=False)
    entities = processor._extract_key_entities(SAMPLE_TEXT)
    
    # Verify that the LLM client was called
    mock_generate_completion.assert_called_once()
    
    # Check entities
    assert isinstance(entities, list)
    assert len(entities) == 5
    assert "AI" in entities
    assert "Machine Learning" in entities


def test_extract_time_periods():
    """Test _extract_time_periods method."""
    processor = SummarizationProcessor()
    
    # Test with dates, years, and quarters
    text = """
    The project started in January 2022. 
    By Q2 2022, we had completed the first phase.
    The second phase will continue until December 31st, 2023.
    We expect to see results in the fourth quarter of 2023.
    The final release is planned for 2024.
    """
    
    periods = processor._extract_time_periods(text)
    
    assert isinstance(periods, list)
    assert len(periods) >= 3
    
    # Check that it found at least some of these patterns
    found_year = False
    found_date = False
    found_quarter = False
    
    for period in periods:
        if re.search(r'2022|2023|2024', period):
            found_year = True
        if re.search(r'January|December', period, re.IGNORECASE):
            found_date = True
        if re.search(r'Q2|fourth quarter', period, re.IGNORECASE):
            found_quarter = True
    
    assert found_year
    assert found_date or found_quarter


def test_extract_json():
    """Test _extract_json method."""
    processor = SummarizationProcessor()
    
    # Test with valid JSON
    text = """
    Here's the evaluation:
    
    {
        "completeness_score": 8.5,
        "conciseness_score": 9.0,
        "accuracy_score": 9.5,
        "overall_score": 9.0
    }
    
    That's my assessment.
    """
    
    json_data = processor._extract_json(text)
    
    assert isinstance(json_data, dict)
    assert len(json_data) == 4
    assert json_data["completeness_score"] == 8.5
    assert json_data["conciseness_score"] == 9.0
    
    # Test with invalid JSON
    text = "This contains no JSON data."
    json_data = processor._extract_json(text)
    assert isinstance(json_data, dict)
    assert len(json_data) == 0


def test_summarize_convenience_function():
    """Test the convenience function 'summarize'."""
    from bytemesumai.summarization.processor import summarize
    
    # Mock the SummarizationProcessor to avoid LLM calls
    with patch('bytemesumai.summarization.processor.SummarizationProcessor') as MockProcessor:
        # Set up the mock to return a SummaryResult
        mock_processor_instance = MagicMock()
        mock_processor_instance.summarize.return_value = SummaryResult(
            summary="Test summary",
            method="basic",
            processing_time=0.5,
            model="gpt-3.5-turbo"
        )
        MockProcessor.return_value = mock_processor_instance
        
        # Call the convenience function
        result = summarize(SAMPLE_TEXT, method="basic", style="concise")
        
        # Verify the processor was instantiated and summarize was called
        MockProcessor.assert_called_once()
        mock_processor_instance.summarize.assert_called_once_with(
            SAMPLE_TEXT, "basic", style="concise"
        )
        
        # Check result
        assert isinstance(result, SummaryResult)
        assert result.summary == "Test summary"
        assert result.method == "basic"


def test_summary_result_to_dict():
    """Test SummaryResult.to_dict method."""
    summary = SummaryResult(
        summary="Test summary",
        method="basic",
        processing_time=0.5,
        model="gpt-3.5-turbo",
        style="concise",
        max_length=100
    )
    
    result_dict = summary.to_dict()
    
    assert isinstance(result_dict, dict)
    assert result_dict["summary"] == "Test summary"
    assert result_dict["method"] == "basic"
    assert result_dict["processing_time"] == 0.5
    assert result_dict["model"] == "gpt-3.5-turbo"
    assert result_dict["style"] == "concise"
    assert result_dict["max_length"] == 100
    assert result_dict["word_count"] == 2  # "Test summary" has 2 words