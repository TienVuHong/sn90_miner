"""
AI-powered agent for statement verification using multiple AI models and data sources.
"""
import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import aiohttp
import structlog

from miner.agents.base_agent import BaseAgent
from miner.agents.resolution_api_client import ResolutionAPIClient
from miner.agents.llm_providers import LLMProviderFactory, LLMProvider
from shared.types import Statement, MinerResponse, Resolution


logger = structlog.get_logger()


class AIAgent(BaseAgent):
    """
    AI-powered agent that uses multiple approaches to verify statements:
    
    1. **Direct API Integration**: Connect to brainstorm/degenbrain resolve endpoint
    2. **Multi-Model AI**: Use OpenAI, Anthropic, or local models for reasoning
    3. **Data Collection**: Fetch real-time data from APIs (CoinGecko, Alpha Vantage, etc.)
    4. **Ensemble Methods**: Combine multiple approaches for better accuracy
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # LLM Provider Configuration
        self.llm_provider_name = config.get("llm_provider", "openai")
        self.llm_provider = LLMProviderFactory.create_provider(
            self.llm_provider_name,
            config,
            timeout=config.get("timeout", 30)
        )
        
        # Brainstorm integration
        self.use_brainstorm = config.get("use_brainstorm", True)
        self.brainstorm_url = config.get("brainstorm_url", "https://degenbrain-459147590380.us-central1.run.app")
        
        # Data Source Configuration
        self.coingecko_api_key = config.get("coingecko_api_key")
        self.alpha_vantage_api_key = config.get("alpha_vantage_api_key")
        
        # Strategy Configuration
        self.strategy = config.get("strategy", "ai_reasoning")
        self.timeout = config.get("timeout", 30)
        
        # Resolution API Configuration
        self.api_url = config.get("api_url", "https://api.subnet90.com")
        self.resolution_client = ResolutionAPIClient(self.api_url, timeout=10)
        
        # Crypto coin mapping cache
        self.coin_lookup = {}
        self.coin_lookup_loaded = False
        
        logger.info("AI Agent initialized", 
                   strategy=self.strategy,
                   llm_provider=self.llm_provider_name,
                   has_llm_provider=bool(self.llm_provider),
                   use_brainstorm=self.use_brainstorm)
    
    async def verify_statement(self, statement_or_synapse) -> MinerResponse:
        """
        Main verification method using AI and data sources.
        """
        # Handle both Statement objects and synapse objects
        if hasattr(statement_or_synapse, 'statement'):
            # This is a synapse object
            statement_text = statement_or_synapse.statement
            statement_id = getattr(statement_or_synapse, 'statement_id', None)
            statement = Statement(
                statement=statement_or_synapse.statement,
                end_date=statement_or_synapse.end_date,
                createdAt=getattr(statement_or_synapse, 'created_at', ''),
                id=statement_id
            )
        else:
            # This is a Statement object
            statement = statement_or_synapse
            statement_text = statement.statement
            statement_id = statement.id
        
        logger.info("Starting AI verification", 
                   statement=statement_text[:60] + "...",
                   strategy=self.strategy,
                   statement_id=statement_id)
        
        try:
            if not self.llm_provider:
                logger.warning("AI reasoning requested but no LLM provider configured")
                return self._create_basic_pending_response(statement)
            return await self._verify_with_ai_reasoning(statement)
                
        except Exception as e:
            logger.error("AI verification failed", error=str(e))
            return self._create_error_response(statement, str(e))
    
    async def _verify_with_brainstorm(self, statement: Statement) -> MinerResponse:
        """
        Use the brainstorm/degenbrain resolve endpoint directly.
        
        This is essentially using the same system as the official validator,
        but miners can use it independently for their own analysis.
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                payload = {
                    "statement": statement.statement,
                    "end_date": statement.end_date,
                    "createdAt": statement.createdAt
                }
                
                async with session.post(f"{self.brainstorm_url}/resolve", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._convert_brainstorm_response(statement, result)
                    else:
                        logger.warning("Brainstorm API error", status=response.status)
                        return await self._verify_with_ai_reasoning(statement)  # Fallback
                        
        except Exception as e:
            logger.error("Brainstorm verification failed", error=str(e))
            return await self._verify_with_ai_reasoning(statement)  # Fallback
    
    async def _verify_with_ai_reasoning(self, statement: Statement) -> MinerResponse:
        """
        Use AI models for reasoning about the statement.
        
        Two-step approach:
        1. Analyze statement to understand what data is needed
        2. Collect data and make final reasoning decision
        """
        # Step 1: Analyze statement
        analysis = await self._analyze_statement(statement)
        
        # Step 2: Collect relevant data
        data = await self._collect_data(analysis)
        
        # Step 3: Make final reasoning decision
        reasoning_result = await self._ai_reasoning(statement, analysis, data)
        
        return reasoning_result
    
    async def _verify_with_resolution_api(self, statement: Statement, statement_id: str) -> Optional[MinerResponse]:
        """
        Use the resolution API to get the official resolution for a statement.
        
        Args:
            statement: Statement to verify
            statement_id: ID of the statement
            
        Returns:
            MinerResponse if resolution found, None otherwise
        """
        try:
            async with self.resolution_client as client:
                api_response = await client.get_resolution(statement_id)
                
                if api_response:
                    # Convert API response to MinerResponse format
                    response_data = client.convert_to_miner_response(api_response, statement.statement)
                    
                    return MinerResponse(
                        statement=response_data["statement"],
                        resolution=Resolution(response_data["resolution"]),
                        confidence=response_data["confidence"],
                        summary=response_data["summary"],
                        sources=response_data["sources"],
                        reasoning=response_data["reasoning"],
                        target_value=response_data.get("target_value"),
                        current_value=response_data.get("current_value")
                    )
                    
        except Exception as e:
            logger.error("Resolution API verification failed", 
                        statement_id=statement_id, 
                        error=str(e))
        
        return None
    
    async def _analyze_statement(self, statement: Statement) -> Dict[str, Any]:
        """
        Use AI to analyze what the statement is asking and what data we need.
        """
        analysis_prompt = f"""
        Analyze this prediction statement and identify:
        1. What type of prediction is this? (price, event, date-based, etc.)
        2. What specific data sources would be needed to verify it?
        3. What is the target value/condition?
        4. What is the deadline?
        
        Statement: {statement.statement}
        End Date: {statement.end_date}
        Initial Value: {statement.initialValue}
        Direction: {statement.direction}
        
        Return your analysis in JSON format:
        {{
            "prediction_type": "...",
            "asset_symbol": "...",
            "target_value": ...,
            "deadline": "...",
            "data_sources_needed": [...],
            "verification_strategy": "..."
        }}
        """
        
        if self.llm_provider:
            return await self.llm_provider.call(analysis_prompt, response_format="json")
        else:
            # Fallback to pattern matching
            return self._pattern_based_analysis(statement)
    
    async def _collect_data(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect relevant data based on the analysis.
        """
        data = {}
        
        # Load coin lookup table first
        await self._load_coin_lookup()
        
        # Crypto price data
        if analysis.get("prediction_type") == "price":
            symbol = analysis.get("asset_symbol")
            deadline = analysis.get("deadline")
            
            if symbol:
                # Try to find the correct coin ID using our lookup
                coin_id = self.coin_lookup.get(symbol.lower(), symbol)
                
                # Check if deadline has passed to determine if we need historical data
                current_time = datetime.now(timezone.utc)
                try:
                    deadline_time = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
                    
                    if deadline_time < current_time:
                        # Deadline has passed - get historical price for that specific date
                        logger.info(f"🕐 Deadline passed, fetching historical price", symbol=symbol, deadline=deadline)
                        data["price_data"] = await self._get_crypto_price(coin_id, deadline)
                        data["verification_type"] = "historical"
                    else:
                        # Deadline in future - get current price
                        logger.info(f"⏰ Deadline in future, fetching current price", symbol=symbol, deadline=deadline)
                        data["price_data"] = await self._get_crypto_price(coin_id)
                        data["verification_type"] = "current"
                        
                except Exception as e:
                    logger.error(f"Error parsing deadline {deadline}: {e}")
                    # Fallback to current price
                    data["price_data"] = await self._get_crypto_price(coin_id)
                    data["verification_type"] = "current"
        
        # Add more data sources as needed
        # data["news_sentiment"] = await self._get_news_sentiment(analysis)
        # data["market_indicators"] = await self._get_market_indicators(analysis)
        
        return data
    
    async def _ai_reasoning(self, statement: Statement, analysis: Dict, data: Dict) -> MinerResponse:
        """
        Use AI to reason about the statement given the analysis and data.
        """
        reasoning_prompt = f"""
        You are a financial prediction verification expert. Based on the data provided, determine if this prediction statement is TRUE, FALSE, or PENDING.
        
        Statement: {statement.statement}
        End Date: {statement.end_date}
        
        Analysis: {json.dumps(analysis, indent=2)}
        Collected Data: {json.dumps(data, indent=2)}
        
        Current Date: {datetime.now(timezone.utc).isoformat()}
        
        CRITICAL VERIFICATION RULES:
        1. If the deadline has passed AND you have historical price data for that date:
           - Check if the actual price on the deadline met the target condition
           - Use the historical price data to determine TRUE/FALSE
           - DO NOT assume FALSE just because the deadline passed
        
        2. If the deadline has passed but NO historical data is available:
           - You can still make an educated assessment based on:
             * General market knowledge about the asset
             * The reasonableness of the target vs current market conditions
             * Your knowledge of the cryptocurrency's historical performance
           - Return PENDING with moderate confidence (30-60%) if unsure
           - Return TRUE/FALSE with lower confidence (20-40%) if you have strong reasoning
           - Always explain your reasoning process and limitations
        
        3. If the deadline is in the future:
           - Return PENDING 
           - Use current price trends for confidence estimation
        
        4. Be precise about price comparisons:
           - "reach $100,000" means >= $100,000
           - "above $4,000" means > $4,000
           - "below $50,000" means < $50,000
        
        5. When data is limited:
           - Lower confidence but still provide your best analysis
           - Explain what additional data would improve your assessment
           - Consider the asset's volatility and market context
        
        Respond in JSON format:
        {{
            "resolution": "TRUE|FALSE|PENDING",
            "confidence": 85,
            "summary": "Detailed explanation of your reasoning and the actual data used...",
            "sources": ["source1", "source2"],
            "key_evidence": "What specific evidence supports this conclusion (include actual prices if available)"
        }}
        """
        
        if self.llm_provider:
            ai_response = await self.llm_provider.call(reasoning_prompt, response_format="json")
        else:
            # Fallback to basic logic
            ai_response = self._basic_reasoning(statement, analysis, data)
        
        return self._convert_ai_response(statement, ai_response)
    
    async def _get_crypto_price(self, symbol: str, date: str = None) -> Dict[str, Any]:
        """Get cryptocurrency price data, optionally for a specific historical date."""
        try:
            headers = {}
            base_url = "https://api.coingecko.com"
            if self.coingecko_api_key:
                headers["x-cg-pro-api-key"] = self.coingecko_api_key
                base_url = "https://pro-api.coingecko.com"  # Use Pro API URL
                logger.info(f"🔑 Using CoinGecko Pro API key for {symbol}")
            
            if date:
                # Try historical price for specific date first
                try:
                    date_obj = datetime.fromisoformat(date.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime('%d-%m-%Y')
                    url = f"{base_url}/api/v3/coins/{symbol}/history?date={formatted_date}"
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=headers) as response:
                            if response.status == 200:
                                result = await response.json()
                                # Extract price from historical data
                                if 'market_data' in result and 'current_price' in result['market_data']:
                                    price_data = {
                                        symbol: {
                                            "usd": result['market_data']['current_price'].get('usd'),
                                            "date": formatted_date,
                                            "historical": True
                                        }
                                    }
                                    logger.info(f"📊 Historical price data", symbol=symbol, date=formatted_date, price=price_data[symbol]["usd"])
                                    return price_data
                                else:
                                    logger.warning(f"No price data in historical response for {symbol} on {formatted_date}, falling back to current price")
                            elif response.status == 429:
                                logger.warning(f"Rate limited on historical data for {symbol}, falling back to current price")
                            else:
                                logger.warning(f"Historical API error {response.status} for {symbol}, falling back to current price")
                except Exception as e:
                    logger.warning(f"Error getting historical data for {symbol}: {e}, falling back to current price")
            
            # Current price (fallback or default)
            url = f"{base_url}/api/v3/simple/price?ids={symbol}&vs_currencies=usd&include_24hr_change=true"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result:
                            # Add metadata to indicate this is current/fallback price
                            for coin_id in result:
                                if isinstance(result[coin_id], dict):
                                    result[coin_id]["historical"] = False
                                    if date:
                                        result[coin_id]["fallback_reason"] = "historical_data_unavailable"
                            logger.info(f"💰 Current price data (fallback)", symbol=symbol, price=result.get(symbol, {}).get("usd"))
                        return result
                    elif response.status == 429:
                        logger.warning(f"Rate limited on current price for {symbol}")
                        # Return empty but don't crash
                        return {}
                    else:
                        logger.warning(f"Current price API error {response.status} for {symbol}")
                        return {}
        except Exception as e:
            logger.error("Failed to get crypto price", symbol=symbol, date=date, error=str(e))
        
        return {}
    
    async def _load_coin_lookup(self) -> None:
        """Load top 100 coins from CoinGecko for symbol/name to ID mapping."""
        if self.coin_lookup_loaded:
            return
            
        try:
            headers = {}
            base_url = "https://api.coingecko.com"
            if self.coingecko_api_key:
                headers["x-cg-pro-api-key"] = self.coingecko_api_key
                base_url = "https://pro-api.coingecko.com"  # Use Pro API URL
            
            url = f"{base_url}/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        coins = await response.json()
                        
                        # Build lookup table: symbol/name -> id
                        for coin in coins:
                            coin_id = coin["id"]
                            symbol = coin["symbol"].lower()
                            name = coin["name"].lower()
                            
                            # Map symbol -> id (e.g., "btc" -> "bitcoin")
                            self.coin_lookup[symbol] = coin_id
                            
                            # Map name -> id (e.g., "bitcoin" -> "bitcoin")
                            self.coin_lookup[name] = coin_id
                            
                            # Map id -> id (e.g., "bitcoin" -> "bitcoin")
                            self.coin_lookup[coin_id] = coin_id
                        
                        self.coin_lookup_loaded = True
                        logger.info(f"Loaded {len(coins)} coins for lookup", total_mappings=len(self.coin_lookup))
                    else:
                        logger.warning(f"Failed to load coin list: {response.status}")
        except Exception as e:
            logger.error("Failed to load coin lookup", error=str(e))
    
    def _find_crypto_symbol(self, text: str) -> Optional[str]:
        """Find a cryptocurrency ID from text using the coin lookup."""
        if not self.coin_lookup_loaded:
            return None
            
        text_lower = text.lower()
        
        # Direct lookups
        for term in self.coin_lookup:
            if term in text_lower:
                return self.coin_lookup[term]
        
        return None
    
    def _pattern_based_analysis(self, statement: Statement) -> Dict[str, Any]:
        """Fallback analysis using pattern matching."""
        statement_text = statement.statement.lower()
        
        # Bitcoin pattern
        if "bitcoin" in statement_text or "btc" in statement_text:
            target_match = re.search(r'\$([0-9,]+)', statement.statement)
            target_value = float(target_match.group(1).replace(',', '')) if target_match else None
            
            return {
                "prediction_type": "crypto_price",
                "asset_symbol": "bitcoin",
                "target_value": target_value,
                "deadline": statement.end_date,
                "data_sources_needed": ["coingecko", "binance"],
                "verification_strategy": "price_comparison"
            }
        
        # Generic fallback
        return {
            "prediction_type": "unknown",
            "verification_strategy": "date_based"
        }
    
    def _basic_reasoning(self, statement: Statement, analysis: Dict, data: Dict) -> Dict[str, Any]:
        """Basic reasoning fallback when AI is not available."""
        # Check if deadline has passed
        try:
            end_date = datetime.fromisoformat(statement.end_date.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            
            if end_date > now:
                return {
                    "resolution": "PENDING",
                    "confidence": 95,
                    "summary": "Deadline has not yet passed",
                    "sources": ["system_clock"],
                    "key_evidence": f"Current time: {now}, Deadline: {end_date}"
                }
            else:
                # For past deadlines, we'd need to check actual data
                # This is a simplified version
                return {
                    "resolution": "FALSE",  # Conservative default
                    "confidence": 30,
                    "summary": "Deadline passed, but insufficient data for verification",
                    "sources": ["basic_analysis"],
                    "key_evidence": "Limited verification capability without AI"
                }
        except:
            return {
                "resolution": "PENDING",
                "confidence": 0,
                "summary": "Error parsing deadline",
                "sources": ["error"],
                "key_evidence": "Could not parse end_date"
            }
    
    def _convert_brainstorm_response(self, statement: Statement, result: Dict) -> MinerResponse:
        """Convert brainstorm API response to MinerResponse format."""
        # Convert string resolution to numeric
        resolution_map = {"TRUE": 1, "FALSE": 0, "PENDING": 2}
        resolution_str = result.get("resolution", "PENDING")
        resolution_int = resolution_map.get(resolution_str, 2)  # Default to PENDING
        
        return MinerResponse(
            statement=statement.statement,
            resolution=Resolution(resolution_int),
            confidence=result.get("confidence", 50),
            summary=result.get("summary", "Brainstorm API response"),
            sources=result.get("sources", ["brainstorm"]),
            target_value=result.get("target_value"),
            current_value=result.get("current_value"),
            reasoning=f"Brainstorm analysis: {result.get('summary', '')}"
        )
    
    def _convert_ai_response(self, statement: Statement, ai_result: Dict) -> MinerResponse:
        """Convert AI reasoning response to MinerResponse format."""
        # Convert string resolution to numeric
        resolution_map = {"TRUE": 1, "FALSE": 0, "PENDING": 2}
        resolution_str = ai_result.get("resolution", "PENDING")
        resolution_int = resolution_map.get(resolution_str, 2)  # Default to PENDING
        
        return MinerResponse(
            statement=statement.statement,
            resolution=Resolution(resolution_int),
            confidence=ai_result.get("confidence", 50),
            summary=ai_result.get("summary", "AI analysis"),
            sources=ai_result.get("sources", ["ai_reasoning"]),
            reasoning=ai_result.get("key_evidence", "AI-powered analysis")
        )
    
    def _create_basic_pending_response(self, statement: Statement) -> MinerResponse:
        """Create a basic pending response when no LLM provider is configured."""
        return MinerResponse(
            statement=statement.statement,
            resolution=Resolution.PENDING,
            confidence=50,
            summary=f"No {self.llm_provider_name} configuration available, unable to verify independently",
            sources=["basic_analysis"],
            reasoning=f"This miner requires a configured LLM provider (set LLM_PROVIDER and corresponding API key) to provide accurate verification"
        )
    
    def _create_error_response(self, statement: Statement, error: str) -> MinerResponse:
        """Create error response when verification fails."""
        return MinerResponse(
            statement=statement.statement,
            resolution=Resolution.PENDING,
            confidence=0,
            summary=f"Verification failed: {error}",
            sources=["error"],
            reasoning=f"Error during AI verification: {error}"
        )