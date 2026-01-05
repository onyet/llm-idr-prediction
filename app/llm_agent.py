"""
Modul LLM Agent untuk Analisis Keuangan Lanjutan
Menyediakan analisis berbasis AI menggunakan berbagai penyedia LLM.

Mendukung:
- OpenAI API
- Ollama (lokal)
- Anthropic Claude
- Penyedia kustom

Berdasarkan arsitektur multi-agent serupa dengan proyek open-source.
"""

import os
import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from .i18n import t

# Muat variabel lingkungan dari file .env jika python-dotenv terpasang
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # Jika dotenv tidak tersedia, lanjutkan tanpa error
    pass


class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    GROQ = "groq"
    MOCK = "mock"  # For testing without actual LLM


@dataclass
class AnalysisContext:
    """Data konteks untuk analisis LLM yang akan diberikan ke agent."""
    pair: str
    current_price: float
    technical_indicators: Dict[str, Any]
    fundamental_data: Dict[str, Any]
    historical_summary: str
    target_date: str
    analysis_type: str  # 'historical' or 'prediction'
    language: str


class BaseLLMAgent(ABC):
    """Kelas dasar untuk semua agent LLM."""
    
    @abstractmethod
    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """Lakukan analisis berdasarkan konteks yang diberikan oleh LLM."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Periksa apakah penyedia LLM tersedia untuk digunakan."""
        pass


class MockLLMAgent(BaseLLMAgent):
    """Agent mock untuk pengujian dan sebagai fallback bila LLM nyata tidak tersedia."""
    
    async def is_available(self) -> bool:
        return True
    
    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """Menghasilkan analisis berbasis aturan sebagai fallback."""
        
        tech = context.technical_indicators
        signal = tech.get("signal", "hold")
        trend = tech.get("trend", "sideways")
        rsi = tech.get("rsi", {}).get("value", 50)
        volatility = tech.get("volatility", 0)
        
        # Buat insight berdasarkan indikator teknikal
        insights = []
        
        # Insight trend
        if trend == "bullish":
            insights.append(self._get_insight("bullish_trend", context.language))
        elif trend == "bearish":
            insights.append(self._get_insight("bearish_trend", context.language))
        else:
            insights.append(self._get_insight("sideways_trend", context.language))
        
        # Insight RSI
        if rsi > 70:
            insights.append(self._get_insight("rsi_high", context.language))
        elif rsi < 30:
            insights.append(self._get_insight("rsi_low", context.language))
        
        # Insight volatilitas
        if volatility > 30:
            insights.append(self._get_insight("high_volatility", context.language))
        elif volatility < 10:
            insights.append(self._get_insight("low_volatility", context.language))
        
        # Buat rekomendasi singkat berdasarkan sinyal
        recommendation = self._generate_recommendation(signal, trend, context.language)
        
        # Penilaian risiko sederhana
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
        """Dapatkan insight yang sudah diterjemahkan menggunakan i18n.
        Kunci terjemahan: `insights.<key>` di module `llm_agent`.
        """
        # Minta terjemahan dari modul i18n
        translation = t(f"insights.{key}", lang, module="llm_agent")
        # Jika tidak ditemukan untuk bahasa yang diminta, fallback ke bahasa Inggris
        if translation == f"insights.{key}":
            translation = t(f"insights.{key}", "en", module="llm_agent")
        return translation
    
    def _generate_recommendation(self, signal: str, trend: str, lang: str) -> str:
        """Dapatkan rekomendasi terjemahan dari modul i18n (kunci: recommendations.<signal>)."""
        key = f"recommendations.{signal}"
        rec = t(key, lang, module="llm_agent")
        if rec == key:
            # fallback ke bahasa Inggris
            rec = t(key, "en", module="llm_agent")
        # Jika masih tidak ditemukan, gunakan rekomendasi 'hold'
        if rec == key:
            rec = t("recommendations.hold", lang, module="llm_agent")
            if rec == "recommendations.hold":
                rec = "Hold"
        return rec
    
    def _generate_summary(self, context: AnalysisContext, trend: str, signal: str) -> str:
        """Bangun ringkasan singkat menggunakan terjemahan dari i18n dan format parameter."""
        current_price = format(context.current_price, ".6f")
        return t(
            "summary.analysis",
            context.language,
            module="llm_agent",
            pair=context.pair,
            target_date=context.target_date,
            trend=trend,
            signal=signal,
            current_price=current_price,
        )


class OllamaAgent(BaseLLMAgent):
    """Agent LLM yang menggunakan Ollama untuk inferensi lokal."""
    
    def __init__(self, model: str = "llama3.2", base_url: Optional[str] = None):
        self.model = model
        import os
        # Jika OLLAMA_BASE_URL diset di .env, gunakan itu; jika tidak gunakan default localhost
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    async def is_available(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            # Jika tidak dapat mengakses Ollama di base_url, anggap tidak tersedia
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
    """Agent LLM menggunakan OpenAI API."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        # API key akan diambil dari environment jika tersedia (dari .env atau sistem)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
    
    async def is_available(self) -> bool:
        # Tersedia jika API key ada
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
        instruction = t("prompt.instruction", context.language, module="llm_agent")

        return f"""Analyze the following market data:

{instruction}

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
    """Agent LLM menggunakan HuggingFace Inference API."""
    
    def __init__(self, api_key: str = None, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        # API key dapat berasal dari HUGGINGFACE_API_KEY atau HF_TOKEN
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        self.model = model
        self.base_url = "https://api-inference.huggingface.co/models"
    
    async def is_available(self) -> bool:
        # Tersedia jika API key ada
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
        instruction = t("prompt.instruction", context.language, module="llm_agent")

        return f"""<s>[INST] You are a professional financial analyst. Analyze the following market data.

{instruction}

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
    """Agent LLM menggunakan Groq API (inference cepat)."""
    
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        # API key diambil dari environment jika tersedia
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
    
    async def is_available(self) -> bool:
        # Tersedia jika API key ada
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
        instruction = t("prompt.instruction", context.language, module="llm_agent")

        return f"""Analyze the following market data:

{instruction}

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
    Anlizer LLM utama yang mengelola beberapa agent dan menyediakan mekanisme fallback.
    """
    
    def __init__(self):
        self.agents: List[BaseLLMAgent] = []
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Inisialisasi agent yang tersedia dalam urutan prioritas."""
        # Coba Groq dulu (cepat, dan ada free tier jika disediakan)
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
