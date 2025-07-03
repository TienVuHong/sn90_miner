"""
Unified LLM provider system for AI agents.
Supports multiple LLM providers with a common interface.
"""
import json
import asyncio
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import aiohttp
import structlog

logger = structlog.get_logger()


class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
    
    @abstractmethod
    async def call(self, prompt: str, response_format: str = "text") -> Any:
        """Make an API call to the LLM provider."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name being used."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider (GPT-4o, GPT-3.5-turbo)."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", timeout: int = 30):
        super().__init__(api_key, timeout)
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    async def call(self, prompt: str, response_format: str = "text") -> Any:
        """Call OpenAI API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial prediction verification expert. Analyze statements accurately and provide structured responses in the requested JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            if response_format == "json":
                payload["response_format"] = {"type": "json_object"}
            
            # Debug logging
            logger.info("🚀 OpenAI API Call", 
                       model=self.model,
                       response_format=response_format,
                       prompt_length=len(prompt))
            logger.debug("📝 OpenAI Prompt", prompt=prompt)
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        
                        # Debug logging
                        logger.info("✅ OpenAI Response", 
                                   status=response.status,
                                   content_length=len(content))
                        logger.debug("📄 OpenAI Response Content", content=content)
                        
                        if response_format == "json":
                            try:
                                parsed_result = json.loads(content)
                                logger.debug("✅ Parsed JSON", parsed_result=parsed_result)
                                return parsed_result
                            except json.JSONDecodeError:
                                logger.error("❌ Failed to parse OpenAI JSON response", content=content)
                                return {"error": "Invalid JSON response from OpenAI"}
                        else:
                            return content
                    else:
                        error_text = await response.text()
                        logger.error("❌ OpenAI API error", status=response.status, error=error_text)
                        return {"error": f"OpenAI API error: {response.status}"}
                        
        except Exception as e:
            logger.error("❌ OpenAI API call failed", error=str(e))
            return {"error": f"OpenAI API call failed: {str(e)}"}
    
    def get_model_name(self) -> str:
        return f"openai/{self.model}"


class AnthropicProvider(LLMProvider):
    """Anthropic provider (Claude 3 Opus, Sonnet, Haiku)."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", timeout: int = 30):
        super().__init__(api_key, timeout)
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
    
    async def call(self, prompt: str, response_format: str = "text") -> Any:
        """Call Anthropic API."""
        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            # Add JSON formatting instruction if needed
            if response_format == "json":
                prompt += "\n\nPlease respond with valid JSON format only."
            
            payload = {
                "model": self.model,
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": f"You are a financial prediction verification expert. Analyze statements accurately and provide structured responses.\n\n{prompt}"
                    }
                ],
                "temperature": 0.1
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["content"][0]["text"]
                        
                        if response_format == "json":
                            try:
                                return json.loads(content)
                            except json.JSONDecodeError:
                                logger.error("Failed to parse Anthropic JSON response", content=content)
                                return {"error": "Invalid JSON response from Anthropic"}
                        else:
                            return content
                    else:
                        error_text = await response.text()
                        logger.error("Anthropic API error", status=response.status, error=error_text)
                        return {"error": f"Anthropic API error: {response.status}"}
                        
        except Exception as e:
            logger.error("Anthropic API call failed", error=str(e))
            return {"error": f"Anthropic API call failed: {str(e)}"}
    
    def get_model_name(self) -> str:
        return f"anthropic/{self.model}"


class GroqProvider(LLMProvider):
    """Groq provider (LLaMA 3 8B, 70B)."""
    
    def __init__(self, api_key: str, model: str = "llama3-70b-8192", timeout: int = 30):
        super().__init__(api_key, timeout)
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    async def call(self, prompt: str, response_format: str = "text") -> Any:
        """Call Groq API (OpenAI-compatible)."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Add JSON formatting instruction if needed
            if response_format == "json":
                prompt += "\n\nPlease respond with valid JSON format only."
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial prediction verification expert. Analyze statements accurately and provide structured responses."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        
                        if response_format == "json":
                            try:
                                return json.loads(content)
                            except json.JSONDecodeError:
                                logger.error("Failed to parse Groq JSON response", content=content)
                                return {"error": "Invalid JSON response from Groq"}
                        else:
                            return content
                    else:
                        error_text = await response.text()
                        logger.error("Groq API error", status=response.status, error=error_text)
                        return {"error": f"Groq API error: {response.status}"}
                        
        except Exception as e:
            logger.error("Groq API call failed", error=str(e))
            return {"error": f"Groq API call failed: {str(e)}"}
    
    def get_model_name(self) -> str:
        return f"groq/{self.model}"


class GeminiProvider(LLMProvider):
    """Google Gemini provider."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro", timeout: int = 30):
        super().__init__(api_key, timeout)
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    async def call(self, prompt: str, response_format: str = "text") -> Any:
        """Call Google Gemini API."""
        try:
            # Add JSON formatting instruction if needed
            if response_format == "json":
                prompt += "\n\nPlease respond with valid JSON format only."
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": f"You are a financial prediction verification expert. Analyze statements accurately and provide structured responses.\n\n{prompt}"
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1000,
                }
            }
            
            params = {"key": self.api_key}
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(self.api_url, params=params, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["candidates"][0]["content"]["parts"][0]["text"]
                        
                        if response_format == "json":
                            try:
                                return json.loads(content)
                            except json.JSONDecodeError:
                                logger.error("Failed to parse Gemini JSON response", content=content)
                                return {"error": "Invalid JSON response from Gemini"}
                        else:
                            return content
                    else:
                        error_text = await response.text()
                        logger.error("Gemini API error", status=response.status, error=error_text)
                        return {"error": f"Gemini API error: {response.status}"}
                        
        except Exception as e:
            logger.error("Gemini API call failed", error=str(e))
            return {"error": f"Gemini API call failed: {str(e)}"}
    
    def get_model_name(self) -> str:
        return f"google/{self.model}"


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider (Mistral, etc.)."""
    
    def __init__(self, api_key: str, model: str = "mistralai/mistral-7b-instruct", timeout: int = 30):
        super().__init__(api_key, timeout)
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    async def call(self, prompt: str, response_format: str = "text") -> Any:
        """Call OpenRouter API (OpenAI-compatible)."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/degenbrain/bittensor-subnet-90",
                "X-Title": "DegenBrain Subnet 90"
            }
            
            # Add JSON formatting instruction if needed
            if response_format == "json":
                prompt += "\n\nPlease respond with valid JSON format only."
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial prediction verification expert. Analyze statements accurately and provide structured responses."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        
                        if response_format == "json":
                            try:
                                return json.loads(content)
                            except json.JSONDecodeError:
                                logger.error("Failed to parse OpenRouter JSON response", content=content)
                                return {"error": "Invalid JSON response from OpenRouter"}
                        else:
                            return content
                    else:
                        error_text = await response.text()
                        logger.error("OpenRouter API error", status=response.status, error=error_text)
                        return {"error": f"OpenRouter API error: {response.status}"}
                        
        except Exception as e:
            logger.error("OpenRouter API call failed", error=str(e))
            return {"error": f"OpenRouter API call failed: {str(e)}"}
    
    def get_model_name(self) -> str:
        return f"openrouter/{self.model}"


class ChutesProvider(LLMProvider):
    """Chutes (Bittensor) provider for decentralized LLM access."""
    
    def __init__(self, cpk_api_key: str, chute_slug: str, model_name: str = "unsloth/Llama-3.2-3B-Instruct", timeout: int = 60):
        # Chutes uses CPK API key and chute slug - increase default timeout for decentralized inference
        super().__init__(cpk_api_key, timeout)
        self.chute_slug = chute_slug
        self.model_name = model_name
        self.api_url = f"https://{chute_slug}.chutes.ai/v1/chat/completions"
    
    async def call(self, prompt: str, response_format: str = "text") -> Any:
        """Call Chutes via HTTP API (OpenAI-compatible)."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Add JSON formatting instruction if needed
            if response_format == "json":
                prompt += "\n\nPlease respond with valid JSON format only."
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial prediction verification expert. Analyze statements accurately and provide structured responses."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000,
                "stream": False  # Don't stream for simplicity
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        
                        if response_format == "json":
                            try:
                                # Chutes sometimes wraps JSON in markdown code blocks - clean it
                                cleaned_content = content.strip()
                                if cleaned_content.startswith("```json"):
                                    cleaned_content = cleaned_content[7:]  # Remove ```json
                                if cleaned_content.startswith("```"):
                                    cleaned_content = cleaned_content[3:]   # Remove ```
                                if cleaned_content.endswith("```"):
                                    cleaned_content = cleaned_content[:-3]  # Remove trailing ```
                                cleaned_content = cleaned_content.strip()
                                
                                parsed_result = json.loads(cleaned_content)
                                logger.debug("✅ Parsed Chutes JSON", parsed_result=parsed_result)
                                return parsed_result
                            except json.JSONDecodeError as e:
                                logger.error("Failed to parse Chutes JSON response", content=content, error=str(e))
                                return {"error": f"Invalid JSON response from Chutes: {str(e)}"}
                        else:
                            return content
                    else:
                        error_text = await response.text()
                        logger.error("Chutes API error", status=response.status, error=error_text)
                        return {"error": f"Chutes API error: {response.status}"}
                        
        except Exception as e:
            logger.error("Chutes API call failed", error=str(e))
            return {"error": f"Chutes API call failed: {str(e)}"}
    
    def get_model_name(self) -> str:
        return f"chutes/{self.model_name}"


class LLMProviderFactory:
    """Factory to create LLM providers based on configuration."""
    
    @staticmethod
    def create_provider(
        provider_name: str,
        config: Dict[str, Any],
        timeout: int = 30
    ) -> Optional[LLMProvider]:
        """Create an LLM provider based on the provider name and config."""
        
        provider_name = provider_name.lower()
        
        if provider_name == "openai":
            api_key = config.get("openai_api_key")
            model = config.get("openai_model", "gpt-4o")
            if not api_key:
                logger.error("OpenAI API key not provided")
                return None
            return OpenAIProvider(api_key, model, timeout)
        
        elif provider_name == "anthropic":
            api_key = config.get("anthropic_api_key")
            model = config.get("anthropic_model", "claude-3-sonnet-20240229")
            if not api_key:
                logger.error("Anthropic API key not provided")
                return None
            return AnthropicProvider(api_key, model, timeout)
        
        elif provider_name == "groq":
            api_key = config.get("groq_api_key")
            model = config.get("groq_model", "llama3-70b-8192")
            if not api_key:
                logger.error("Groq API key not provided")
                return None
            return GroqProvider(api_key, model, timeout)
        
        elif provider_name == "gemini":
            api_key = config.get("gemini_api_key")
            model = config.get("gemini_model", "gemini-1.5-pro")
            if not api_key:
                logger.error("Gemini API key not provided")
                return None
            return GeminiProvider(api_key, model, timeout)
        
        elif provider_name == "openrouter":
            api_key = config.get("openrouter_api_key")
            model = config.get("openrouter_model", "mistralai/mistral-7b-instruct")
            if not api_key:
                logger.error("OpenRouter API key not provided")
                return None
            return OpenRouterProvider(api_key, model, timeout)
        
        elif provider_name == "chutes":
            cpk_api_key = config.get("chutes_cpk_api_key")
            chute_slug = config.get("chutes_slug")
            model_name = config.get("chutes_model", "unsloth/Llama-3.2-3B-Instruct")
            # Use longer timeout for Chutes decentralized inference
            chutes_timeout = max(timeout, 60)  # At least 60 seconds for Chutes
            if not cpk_api_key or not chute_slug:
                logger.error("Chutes requires chutes_cpk_api_key and chutes_slug")
                return None
            return ChutesProvider(cpk_api_key, chute_slug, model_name, chutes_timeout)
        
        else:
            logger.error(f"Unknown provider: {provider_name}")
            return None