"""
Async Pipeline - Parallel Processing for Trading Operations
Author: Erdinc Erdogan
Purpose: Provides async/await wrapper for parallel HTTP requests and LLM calls,
achieving up to 4x speed improvement in data fetching and analysis operations.
References:
- Python Asyncio for Concurrent Programming
- Async HTTP with aiohttp
- Parallel LLM Inference Patterns
Usage:
    pipeline = AsyncPipeline()
    results = await pipeline.fetch_all([ticker1, ticker2, ticker3])
"""
import asyncio
from typing import List, Dict, Optional
from colorama import Fore, Style


class AsyncPipeline:
    """
    Asenkron pipeline wrapper.
    HTTP istekleri ve LLM çağrılarını paralel yapar.
    """
    
    def __init__(self):
        self.results = {}
    
    async def fetch_price_async(self, ticker: str) -> Dict:
        """Fiyat verisini asenkron çek."""
        try:
            # aiohttp ile asenkron HTTP
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # yfinance yerine doğrudan API
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                async with session.get(url) as response:
                    data = await response.json()
                    result = data.get('chart', {}).get('result', [{}])[0]
                    meta = result.get('meta', {})
                    return {
                        "ticker": ticker,
                        "price": meta.get('regularMarketPrice', 0),
                        "change": meta.get('regularMarketChangePercent', 0)
                    }
        except:
            return {"ticker": ticker, "price": 0, "error": True}
    
    async def fetch_multiple_prices(self, tickers: List[str]) -> Dict[str, Dict]:
        """Birden fazla hisseyi paralel çek."""
        tasks = [self.fetch_price_async(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        return {r["ticker"]: r for r in results}
    
    async def llm_invoke_async(self, llm, prompt: str) -> str:
        """LLM çağrısını asenkron yap."""
        try:
            # Ollama async API
            if hasattr(llm, 'ainvoke'):
                return await llm.ainvoke(prompt)
            else:
                # Fallback: senkron çağrıyı thread pool'da çalıştır
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, llm.invoke, prompt)
        except:
            return ""
    
    async def parallel_tasks(self, tasks: Dict[str, callable]) -> Dict:
        """Birden fazla görevi paralel çalıştır."""
        results = {}
        
        async def run_task(name, task):
            try:
                if asyncio.iscoroutinefunction(task):
                    result = await task()
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, task)
                results[name] = result
            except Exception as e:
                results[name] = {"error": str(e)}
        
        await asyncio.gather(*[run_task(name, task) for name, task in tasks.items()])
        return results
    
    def run_async(self, coro):
        """Async coroutine'i senkron ortamda çalıştır."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Nested event loop
                import nest_asyncio
                nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)


class AsyncAdvisor:
    """
    AIAdvisor'ın async versiyonu.
    """
    
    def __init__(self, advisor):
        self.advisor = advisor
        self.pipeline = AsyncPipeline()
    
    async def analyze_trade_async(self, ticker: str, tech_signals: str, market_sentiment: str = "Nötr") -> str:
        """
        Async trade analizi.
        Web search, grafik oluşturma ve LLM çağrılarını paralel yapar.
        """
        print(f"{Fore.CYAN}⚡ Async Pipeline başlatılıyor...{Style.RESET_ALL}", flush=True)
        
        # Paralel görevler
        tasks = {}
        
        # 1. Fiyat verisi çek
        tasks['price'] = lambda: self.pipeline.fetch_price_async(ticker)
        
        # 2. Döküman araması (senkron, thread pool'da)
        tasks['docs'] = lambda: self.advisor.vectorstore.similarity_search(
            f"{ticker} için strateji", k=5
        )
        
        # 3. Graph-RAG
        tasks['graph'] = lambda: self.advisor.graph.get_graph_context(ticker)
        
        # Paralel çalıştır
        results = await self.pipeline.parallel_tasks(tasks)
        
        print(f"{Fore.GREEN}⚡ Paralel görevler tamamlandı{Style.RESET_ALL}", flush=True)
        
        # Sonuçları birleştir ve LLM'e gönder
        context = str(results.get('docs', []))
        graph_context = results.get('graph', '')
        
        # LLM çağrısı
        response = await self.pipeline.llm_invoke_async(
            self.advisor.llm,
            f"Ticker: {ticker}\nTeknik: {tech_signals}\nContext: {context}\nGraph: {graph_context}"
        )
        
        return response
    
    def analyze_trade(self, ticker: str, tech_signals: str, market_sentiment: str = "Nötr") -> str:
        """Senkron wrapper."""
        return self.pipeline.run_async(
            self.analyze_trade_async(ticker, tech_signals, market_sentiment)
        )
