"""
Provider-agnostic LLM client using LiteLLM for SumWiseAI.

This module provides a unified interface for interacting with different LLM
providers through LiteLLM, with additional error handling and caching.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

import litellm
from litellm import completion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMClient:
    """
    Provider-agnostic LLM client using LiteLLM.
    
    This client handles different providers, error handling, retries, 
    and can be extended with caching.
    """
    
    def __init__(
        self, 
        model: str = "gpt-3.5-turbo", 
        api_key: Optional[str] = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        temperature: float = 0.0,
        cache_enabled: bool = False
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: Model identifier (can be from any supported provider)
            api_key: Optional API key (otherwise uses environment variables)
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Base delay between retries (in seconds)
            temperature: Default temperature for generation (0.0 = deterministic)
            cache_enabled: Whether to enable response caching
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.cache_enabled = cache_enabled
        self._cache = {}  # Simple in-memory cache
        
        # Set API key if provided
        if api_key:
            # Set appropriate provider key based on model prefix
            if model.startswith(("gpt", "text-embedding")):
                os.environ["OPENAI_API_KEY"] = api_key
            elif model.startswith("claude"):
                os.environ["ANTHROPIC_API_KEY"] = api_key
            # Add more provider-specific keys as needed
        
        logger.info(f"Initialized LLMClient with model {model}")
    
    def generate_completion(
        self, 
        prompt: str, 
        system_message: Optional[str] = None, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate completion using the configured LLM.
        
        Args:
            prompt: Prompt text or user message
            system_message: Optional system message
            max_tokens: Maximum tokens in the response
            temperature: Temperature for generation (overrides instance default)
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Generated text response
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Check cache if enabled
        cache_key = self._get_cache_key(messages, max_tokens, temperature)
        if self.cache_enabled and cache_key in self._cache:
            logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
            return self._cache[cache_key]
        
        # Set parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        
        # Add max_tokens if specified
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        # Add any additional parameters
        params.update(kwargs)
        
        # Try to generate with retries
        for attempt in range(self.max_retries + 1):
            try:
                response = completion(**params)
                content = response.choices[0].message.content
                
                # Cache the response if caching is enabled
                if self.cache_enabled:
                    self._cache[cache_key] = content
                
                return content
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {str(e)}")
                
                # Check if this is the last attempt
                if attempt < self.max_retries:
                    # Determine if we should retry based on error type
                    if self._should_retry(e):
                        # Exponential backoff
                        delay = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                        
                        # Try fallback to a more reliable model if appropriate
                        if attempt == self.max_retries - 1 and "context length" in str(e).lower():
                            logger.info("Context length issue detected, trying to truncate prompt...")
                            # Truncate prompt to try to fit context window
                            prompt_words = prompt.split()
                            if len(prompt_words) > 100:
                                truncated_prompt = " ".join(prompt_words[:int(len(prompt_words) * 0.75)])
                                messages = [{"role": "user", "content": truncated_prompt}]
                                if system_message:
                                    messages.insert(0, {"role": "system", "content": system_message})
                                params["messages"] = messages
                    else:
                        # Error not recoverable with retries
                        break
                else:
                    # Last attempt failed
                    break
        
        # If we reached here, all attempts failed
        error_msg = f"Failed to generate completion after {self.max_retries + 1} attempts"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def _should_retry(self, exception: Exception) -> bool:
        """
        Determine if a request should be retried based on the exception.
        
        Args:
            exception: The exception raised by the request
            
        Returns:
            Whether to retry the request
        """
        error_message = str(exception).lower()
        
        # Common retryable errors
        retryable_patterns = [
            "rate limit",
            "timeout",
            "connection",
            "server",
            "too many requests",
            "capacity",
            "overloaded",
            "temporarily unavailable"
        ]
        
        return any(pattern in error_message for pattern in retryable_patterns)
    
    def _get_cache_key(self, messages: List[Dict], max_tokens: Optional[int], temperature: Optional[float]) -> str:
        """
        Generate a cache key for the given request parameters.
        
        Args:
            messages: The messages to be sent
            max_tokens: Maximum tokens parameter
            temperature: Temperature parameter
            
        Returns:
            Cache key string
        """
        # Create a simple representation of messages
        message_str = str([(m.get("role", ""), m.get("content", "")[:100]) for m in messages])
        
        # Combine with other parameters that affect the output
        return f"{self.model}:{message_str}:{max_tokens}:{temperature or self.temperature}"