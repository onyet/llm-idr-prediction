"""
LLM Agent Module for Advanced Financial Analysis
Provides AI-powered analysis using various LLM providers.

Supports:
- OpenAI API
- Ollama (local)
- Anthropic Claude
- Custom providers

Based on multi-agent architecture similar to:
https://github.com/imanoop7/Financial-Analysis--Multi-Agent-Open-Source-LLM
"""

import os
import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    GROQ = "groq"
    MOCK = "mock"  # For testing without actual LLM


@dataclass
class AnalysisContext:
    """Context data for LLM analysis."""
    pair: str
    current_price: float
    technical_indicators: Dict[str, Any]
    fundamental_data: Dict[str, Any]
    historical_summary: str
    target_date: str
    analysis_type: str  # 'historical' or 'prediction'
    language: str


class BaseLLMAgent(ABC):
    """Base class for LLM agents."""
    
    @abstractmethod
    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """Perform analysis using LLM."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the LLM provider is available."""
        pass


class MockLLMAgent(BaseLLMAgent):
    """Mock LLM agent for testing and fallback."""
    
    async def is_available(self) -> bool:
        return True
    
    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """Generate rule-based analysis as fallback."""
        
        tech = context.technical_indicators
        signal = tech.get("signal", "hold")
        trend = tech.get("trend", "sideways")
        rsi = tech.get("rsi", {}).get("value", 50)
        volatility = tech.get("volatility", 0)
        
        # Generate insights based on indicators
        insights = []
        
        # Trend insight
        if trend == "bullish":
            insights.append(self._get_insight("bullish_trend", context.language))
        elif trend == "bearish":
            insights.append(self._get_insight("bearish_trend", context.language))
        else:
            insights.append(self._get_insight("sideways_trend", context.language))
        
        # RSI insight
        if rsi > 70:
            insights.append(self._get_insight("rsi_high", context.language))
        elif rsi < 30:
            insights.append(self._get_insight("rsi_low", context.language))
        
        # Volatility insight
        if volatility > 30:
            insights.append(self._get_insight("high_volatility", context.language))
        elif volatility < 10:
            insights.append(self._get_insight("low_volatility", context.language))
        
        # Generate recommendation
        recommendation = self._generate_recommendation(signal, trend, context.language)
        
        # Risk assessment
        risk_level = "high" if volatility > 25 or abs(rsi - 50) > 25 else "medium" if volatility > 15 else "low"
        
        return {
            "provider": "rule_based",
            "insights": insights,
            "recommendation": recommendation,
            "risk_level": risk_level,
            "confidence": 0.7,  # Fixed confidence for rule-based
            "summary": self._generate_summary(context, trend, signal),
        }
    
    def _get_insight(self, key: str, lang: str) -> str:
        insights = {
            "bullish_trend": {
                "id": "Tren saat ini menunjukkan momentum bullish yang kuat",
                "en": "Current trend shows strong bullish momentum",
                "ar": "الاتجاه الحالي يُظهر زخمًا صعوديًا قويًا"
            },
            "bearish_trend": {
                "id": "Tren saat ini menunjukkan tekanan bearish",
                "en": "Current trend shows bearish pressure",
                "ar": "الاتجاه الحالي يُظهر ضغطًا هبوطيًا"
            },
            "sideways_trend": {
                "id": "Pasar sedang dalam fase konsolidasi",
                "en": "Market is in consolidation phase",
                "ar": "السوق في مرحلة تماسك"
            },
            "rsi_high": {
                "id": "Indikator RSI menunjukkan kondisi overbought, waspadai koreksi",
                "en": "RSI indicator shows overbought condition, watch for correction",
                "ar": "مؤشر RSI يُظهر حالة ذروة الشراء، احذر من التصحيح"
            },
            "rsi_low": {
                "id": "Indikator RSI menunjukkan kondisi oversold, potensi rebound",
                "en": "RSI indicator shows oversold condition, potential rebound",
                "ar": "مؤشر RSI يُظهر حالة ذروة البيع، احتمال الارتداد"
            },
            "high_volatility": {
                "id": "Volatilitas tinggi mengindikasikan risiko yang lebih besar",
                "en": "High volatility indicates greater risk",
                "ar": "التقلب العالي يشير إلى مخاطر أكبر"
            },
            "low_volatility": {
                "id": "Volatilitas rendah menunjukkan stabilitas pasar",
                "en": "Low volatility shows market stability",
                "ar": "التقلب المنخفض يُظهر استقرار السوق"
            }
        }
        return insights.get(key, {}).get(lang, insights.get(key, {}).get("en", key))
    
    def _generate_recommendation(self, signal: str, trend: str, lang: str) -> str:
        recommendations = {
            "strong_buy": {
                "id": "Sinyal sangat positif untuk posisi beli",
                "en": "Very positive signal for buy position",
                "ar": "إشارة إيجابية جدًا لموقف الشراء"
            },
            "buy": {
                "id": "Pertimbangkan untuk membuka posisi beli dengan manajemen risiko yang baik",
                "en": "Consider opening buy position with proper risk management",
                "ar": "فكر في فتح موقف شراء مع إدارة مخاطر مناسبة"
            },
            "hold": {
                "id": "Pertahankan posisi saat ini dan tunggu konfirmasi lebih lanjut",
                "en": "Maintain current position and wait for further confirmation",
                "ar": "حافظ على الموقف الحالي وانتظر مزيدًا من التأكيد"
            },
            "sell": {
                "id": "Pertimbangkan untuk mengambil profit atau mengurangi eksposur",
                "en": "Consider taking profit or reducing exposure",
                "ar": "فكر في جني الأرباح أو تقليل التعرض"
            },
            "strong_sell": {
                "id": "Sinyal negatif kuat, pertimbangkan untuk keluar dari posisi",
                "en": "Strong negative signal, consider exiting position",
                "ar": "إشارة سلبية قوية، فكر في الخروج من الموقف"
            }
        }
        return recommendations.get(signal, {}).get(lang, recommendations.get(signal, {}).get("en", "Hold"))
    
    def _generate_summary(self, context: AnalysisContext, trend: str, signal: str) -> str:
        summaries = {
            "id": f"Analisis {context.pair} untuk {context.target_date}: Tren {trend}, sinyal {signal}. "
                  f"Harga saat ini {context.current_price:.6f}.",
            "en": f"Analysis of {context.pair} for {context.target_date}: Trend {trend}, signal {signal}. "
                  f"Current price {context.current_price:.6f}.",
            "ar": f"تحليل {context.pair} لـ {context.target_date}: الاتجاه {trend}، الإشارة {signal}. "
                  f"السعر الحالي {context.current_price:.6f}."
        }
        return summaries.get(context.language, summaries["en"])


class OllamaAgent(BaseLLMAgent):
    """LLM agent using Ollama for local inference."""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    async def is_available(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        try:
            import httpx
            
            prompt = self._build_prompt(context)
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    analysis_text = result.get("response", "")
                    
                    # Try to parse JSON response
                    try:
                        parsed = json.loads(analysis_text)
                        return {
                            "provider": f"ollama/{self.model}",
                            **parsed
                        }
                    except json.JSONDecodeError:
                        return {
                            "provider": f"ollama/{self.model}",
                            "insights": [analysis_text],
                            "recommendation": "See detailed analysis",
                            "risk_level": "medium",
                            "confidence": 0.6,
                            "summary": analysis_text[:500]
                        }
                else:
                    raise Exception(f"Ollama API error: {response.status_code}")
                    
        except Exception as e:
            # Fallback to mock
            mock = MockLLMAgent()
            result = await mock.analyze(context)
            result["provider"] = f"fallback (ollama error: {str(e)[:50]})"
            return result
    
    def _build_prompt(self, context: AnalysisContext) -> str:
        lang_instruction = {
            "id": "Berikan analisis dalam Bahasa Indonesia.",
            "en": "Provide analysis in English.",
            "ar": "قدم التحليل باللغة العربية."
        }
        
        return f"""You are a professional financial analyst. Analyze the following market data and provide insights.

{lang_instruction.get(context.language, lang_instruction['en'])}

Market Data:
- Pair: {context.pair}
- Current Price: {context.current_price}
- Analysis Type: {context.analysis_type}
- Target Date: {context.target_date}

Technical Indicators:
{json.dumps(context.technical_indicators, indent=2)}

Fundamental Data:
{json.dumps(context.fundamental_data, indent=2)}

Historical Summary:
{context.historical_summary}

Provide your analysis in the following JSON format:
{{
    "insights": ["insight1", "insight2", ...],
    "recommendation": "your recommendation",
    "risk_level": "low|medium|high",
    "confidence": 0.0-1.0,
    "summary": "brief summary"
}}
"""


class OpenAIAgent(BaseLLMAgent):
    """LLM agent using OpenAI API."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
    
    async def is_available(self) -> bool:
        return bool(self.api_key)
    
    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        if not self.api_key:
            mock = MockLLMAgent()
            result = await mock.analyze(context)
            result["provider"] = "fallback (no OpenAI API key)"
            return result
        
        try:
            import httpx
            
            prompt = self._build_prompt(context)
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a professional financial analyst. Always respond in valid JSON format."},
                            {"role": "user", "content": prompt}
                        ],
                        "response_format": {"type": "json_object"}
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    return {
                        "provider": f"openai/{self.model}",
                        **parsed
                    }
                else:
                    raise Exception(f"OpenAI API error: {response.status_code}")
                    
        except Exception as e:
            mock = MockLLMAgent()
            result = await mock.analyze(context)
            result["provider"] = f"fallback (openai error: {str(e)[:50]})"
            return result
    
    def _build_prompt(self, context: AnalysisContext) -> str:
        lang_instruction = {
            "id": "Berikan analisis dalam Bahasa Indonesia.",
            "en": "Provide analysis in English.",
            "ar": "قدم التحليل باللغة العربية."
        }
        
        return f"""Analyze the following market data:

{lang_instruction.get(context.language, lang_instruction['en'])}

Market Data:
- Pair: {context.pair}
- Current Price: {context.current_price}
- Analysis Type: {context.analysis_type}
- Target Date: {context.target_date}

Technical Indicators:
{json.dumps(context.technical_indicators, indent=2)}

Fundamental Data:
{json.dumps(context.fundamental_data, indent=2)}

Historical Summary:
{context.historical_summary}

Respond with JSON containing: insights (array), recommendation (string), risk_level (low/medium/high), confidence (0-1), summary (string)
"""


class HuggingFaceAgent(BaseLLMAgent):
    """LLM agent using HuggingFace Inference API."""
    
    def __init__(self, api_key: str = None, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        self.model = model
        self.base_url = "https://api-inference.huggingface.co/models"
    
    async def is_available(self) -> bool:
        return bool(self.api_key)
    
    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        if not self.api_key:
            mock = MockLLMAgent()
            result = await mock.analyze(context)
            result["provider"] = "fallback (no HuggingFace API key)"
            return result
        
        try:
            import httpx
            
            prompt = self._build_prompt(context)
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/{self.model}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 1024,
                            "temperature": 0.7,
                            "return_full_text": False
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                    else:
                        generated_text = str(result)
                    
                    # Try to extract JSON from response
                    try:
                        # Find JSON in response
                        import re
                        json_match = re.search(r'\{[^{}]*\}', generated_text, re.DOTALL)
                        if json_match:
                            parsed = json.loads(json_match.group())
                            return {
                                "provider": f"huggingface/{self.model.split('/')[-1]}",
                                **parsed
                            }
                    except (json.JSONDecodeError, AttributeError):
                        pass
                    
                    # Fallback: parse text response
                    return {
                        "provider": f"huggingface/{self.model.split('/')[-1]}",
                        "insights": [generated_text[:500]],
                        "recommendation": "See detailed analysis",
                        "risk_level": "medium",
                        "confidence": 0.6,
                        "summary": generated_text[:300]
                    }
                else:
                    raise Exception(f"HuggingFace API error: {response.status_code}")
                    
        except Exception as e:
            mock = MockLLMAgent()
            result = await mock.analyze(context)
            result["provider"] = f"fallback (huggingface error: {str(e)[:50]})"
            return result
    
    def _build_prompt(self, context: AnalysisContext) -> str:
        lang_instruction = {
            "id": "Berikan analisis dalam Bahasa Indonesia.",
            "en": "Provide analysis in English.",
            "ar": "قدم التحليل باللغة العربية."
        }
        
        return f"""<s>[INST] You are a professional financial analyst. Analyze the following market data.

{lang_instruction.get(context.language, lang_instruction['en'])}

Market Data:
- Pair: {context.pair}
- Current Price: {context.current_price}
- Analysis Type: {context.analysis_type}
- Target Date: {context.target_date}

Technical Indicators:
{json.dumps(context.technical_indicators, indent=2)}

Fundamental Data:
{json.dumps(context.fundamental_data, indent=2)}

Historical Summary:
{context.historical_summary}

Respond with JSON: {{"insights": [...], "recommendation": "...", "risk_level": "low|medium|high", "confidence": 0.0-1.0, "summary": "..."}} [/INST]"""


class GroqAgent(BaseLLMAgent):
    """LLM agent using Groq API (fast inference)."""
    
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
    
    async def is_available(self) -> bool:
        return bool(self.api_key)
    
    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        if not self.api_key:
            mock = MockLLMAgent()
            result = await mock.analyze(context)
            result["provider"] = "fallback (no Groq API key)"
            return result
        
        try:
            import httpx
            
            prompt = self._build_prompt(context)
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a professional financial analyst. Always respond in valid JSON format."},
                            {"role": "user", "content": prompt}
                        ],
                        "response_format": {"type": "json_object"}
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    return {
                        "provider": f"groq/{self.model}",
                        **parsed
                    }
                else:
                    raise Exception(f"Groq API error: {response.status_code}")
                    
        except Exception as e:
            mock = MockLLMAgent()
            result = await mock.analyze(context)
            result["provider"] = f"fallback (groq error: {str(e)[:50]})"
            return result
    
    def _build_prompt(self, context: AnalysisContext) -> str:
        lang_instruction = {
            "id": "Berikan analisis dalam Bahasa Indonesia.",
            "en": "Provide analysis in English.",
            "ar": "قدم التحليل باللغة العربية."
        }
        
        return f"""Analyze the following market data:

{lang_instruction.get(context.language, lang_instruction['en'])}

Market Data:
- Pair: {context.pair}
- Current Price: {context.current_price}
- Analysis Type: {context.analysis_type}
- Target Date: {context.target_date}

Technical Indicators:
{json.dumps(context.technical_indicators, indent=2)}

Fundamental Data:
{json.dumps(context.fundamental_data, indent=2)}

Historical Summary:
{context.historical_summary}

Respond with JSON containing: insights (array), recommendation (string), risk_level (low/medium/high), confidence (0-1), summary (string)
"""


class LLMAnalyzer:
    """
    Main LLM analyzer that manages multiple agents and provides fallback.
    """
    
    def __init__(self):
        self.agents: List[BaseLLMAgent] = []
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize available agents in priority order."""
        # Try Groq first (fast, free tier available)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            self.agents.append(GroqAgent(api_key=groq_key))
        
        # Try Ollama (local, free)
        self.agents.append(OllamaAgent())
        
        # Then HuggingFace (free tier available)
        hf_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        if hf_key:
            self.agents.append(HuggingFaceAgent(api_key=hf_key))
        
        # Then OpenAI (requires API key)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.agents.append(OpenAIAgent(api_key=openai_key))
        
        # Always have mock as fallback
        self.agents.append(MockLLMAgent())
    
    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """
        Perform analysis using the first available agent.
        Falls back to next agent if one fails.
        """
        for agent in self.agents:
            if await agent.is_available():
                try:
                    return await agent.analyze(context)
                except Exception as e:
                    continue  # Try next agent
        
        # If all else fails, use mock
        mock = MockLLMAgent()
        return await mock.analyze(context)
    
    async def multi_agent_analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """
        Run analysis with multiple agents and combine results.
        Useful for getting diverse perspectives.
        """
        results = []
        
        for agent in self.agents:
            if await agent.is_available():
                try:
                    result = await agent.analyze(context)
                    results.append(result)
                except Exception:
                    continue
        
        if not results:
            mock = MockLLMAgent()
            return await mock.analyze(context)
        
        # Combine insights from all agents
        all_insights = []
        recommendations = []
        risk_levels = []
        confidences = []
        
        for r in results:
            all_insights.extend(r.get("insights", []))
            recommendations.append(r.get("recommendation", ""))
            risk_levels.append(r.get("risk_level", "medium"))
            confidences.append(r.get("confidence", 0.5))
        
        # Deduplicate insights
        unique_insights = list(dict.fromkeys(all_insights))
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Most common risk level
        risk_count = {}
        for r in risk_levels:
            risk_count[r] = risk_count.get(r, 0) + 1
        consensus_risk = max(risk_count, key=risk_count.get)
        
        return {
            "provider": "multi_agent",
            "agents_used": [r.get("provider", "unknown") for r in results],
            "insights": unique_insights[:10],  # Limit to 10 insights
            "recommendations": recommendations,
            "risk_level": consensus_risk,
            "confidence": round(avg_confidence, 2),
            "summary": results[0].get("summary", "") if results else ""
        }


# Singleton instance
_llm_analyzer: Optional[LLMAnalyzer] = None


def get_llm_analyzer() -> LLMAnalyzer:
    """Get or create the LLM analyzer instance."""
    global _llm_analyzer
    if _llm_analyzer is None:
        _llm_analyzer = LLMAnalyzer()
    return _llm_analyzer
