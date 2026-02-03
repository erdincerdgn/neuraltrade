"""
Model Orchestrator for Multi-LLM Routing
Author: Erdinc Erdogan
Purpose: Routes queries to optimal LLM (Ollama local, GPT-4, Vision models) based on complexity, image presence, and cost/latency requirements.
References:
- LLM Routing Patterns
- Multi-Model Orchestration
- Cost-Aware Model Selection
Usage:
    orchestrator = ModelOrchestrator(ollama_host="http://localhost:11434")
    response = orchestrator.invoke(prompt, model_config=orchestrator.select_model(query, complexity))
"""
import os
from typing import Dict
from colorama import Fore, Style
from langchain_ollama import OllamaLLM


class ModelOrchestrator:
    """Sorgunun tipine göre en uygun modeli seçer."""
    
    MODELS = {
        "fast": {"name": "llama3", "provider": "ollama", "desc": "⚡ Hızlı"},
        "balanced": {"name": "llama3:70b", "provider": "ollama", "desc": "⚖️ Dengeli"},
        "premium": {"name": "gpt-4o-mini", "provider": "openai", "desc": "🏆 Premium"},
        "vision": {"name": "llama3.2-vision", "provider": "ollama", "desc": "👁️ Vision"},
    }
    
    def __init__(self, ollama_host: str, openai_key: str = None):
        self.ollama_host = ollama_host
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.default_model = "fast"
    
    def select_model(self, query: str, complexity, has_image: bool = False) -> Dict:
        """Sorguya göre model seç."""
        from ..core.base import QueryComplexity
        
        if has_image:
            return self.MODELS["vision"]
        
        if complexity == QueryComplexity.SIMPLE:
            return self.MODELS["fast"]
        elif complexity == QueryComplexity.COMPLEX:
            if self.openai_key:
                return self.MODELS["premium"]
            return self.MODELS["balanced"]
        
        return self.MODELS["fast"]
    
    def invoke(self, prompt: str, model_config: Dict) -> str:
        """Seçilen model ile çağrı yap."""
        if model_config["provider"] == "ollama":
            llm = OllamaLLM(model=model_config["name"], base_url=self.ollama_host)
            return llm.invoke(prompt)
        
        elif model_config["provider"] == "openai" and self.openai_key:
            try:
                import urllib.request
                import json
                
                payload = json.dumps({
                    "model": model_config["name"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000
                }).encode('utf-8')
                
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=payload,
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {self.openai_key}'
                    }
                )
                
                with urllib.request.urlopen(req, timeout=30) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    return result["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"{Fore.YELLOW}  → OpenAI hata: {e}{Style.RESET_ALL}", flush=True)
        
        llm = OllamaLLM(model="llama3", base_url=self.ollama_host)
        return llm.invoke(prompt)
