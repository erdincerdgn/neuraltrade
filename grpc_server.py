import os
import sys
import logging
import time
from concurrent import futures
import grpc

# Yol ayarları
current_dir = os.path.dirname(os.path.abspath(__file__))
generated_path = os.path.join(current_dir, 'proto', 'generated')
sys.path.insert(0, generated_path)

import ai_service_pb2
import ai_service_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIServiceServicer(ai_service_pb2_grpc.AIServiceServicer):
    
    def __init__(self):
        self.start_time = time.time()
        self._initialized = False
        self.pipeline = None

    def _init_if_needed(self):
        if self._initialized: return
        try:
            from modules.data.data_loader import get_market_data_bundle
            from modules.alpha.technical_analysis import analyze_data
            from modules.core.ai_advisor import AIAdvisor
            self.pipeline = {
                "data_loader": get_market_data_bundle,
                "analyzer": analyze_data,
                "advisor": AIAdvisor()
            }
            self._initialized = True
            logger.info("✅ AI Engine Pipeline Ready")
        except Exception as e:
            logger.error(f"❌ Pipeline Init Failed: {e}")

    # --- PredictSignal ---
    def PredictSignal(self, request, context):
        self._init_if_needed()
        response = ai_service_pb2.SignalResponse()
        try:
            symbol = request.symbol
            if self.pipeline:
                data = self.pipeline["data_loader"](symbol)
                analysis = self.pipeline["analyzer"](data)
                signal = self.pipeline["advisor"].get_signal(symbol, data, analysis)
                
                response.action = str(signal.get("action", "HOLD"))
                response.confidence = float(signal.get("confidence", 0.5))
                response.reasoning = str(signal.get("reasoning", "Success"))
            else:
                response.reasoning = "Pipeline not ready"
        except Exception as e:
            response.reasoning = f"Error: {str(e)}"
        return response

    # --- SelectModel (Proto: ModelSelectionResponse) ---
    def SelectModel(self, request, context):
        # ⚠️ KRİTİK: Proto'daki isim 'ModelSelectionResponse'
        response = ai_service_pb2.ModelSelectionResponse()
        response.model_id = "ensemble_v2"
        response.model_name = "NeuralTrade Ensemble"
        response.confidence = 0.92
        response.reasoning = f"Optimal for {request.regime}"
        return response

    # --- RouteStrategy (Proto: StrategyResponse) ---
    def RouteStrategy(self, request, context):
        response = ai_service_pb2.StrategyResponse()
        response.strategy_id = "trend_follower_v1"
        response.strategy_name = "Smart Trend Follower"
        response.execution_type = "NEUTRAL"
        response.position_size_modifier = 1.0
        return response

    # --- AnalyzeSentiment (Proto: SentimentResponse) ---
    def AnalyzeSentiment(self, request, context):
        response = ai_service_pb2.SentimentResponse()
        response.overall_sentiment = 0.45
        response.bullish_ratio = 0.60
        response.bearish_ratio = 0.20
        response.neutral_ratio = 0.20
        return response

    # --- HealthCheck (Proto: HealthResponse) ---
    def HealthCheck(self, request, context):
        response = ai_service_pb2.HealthResponse()
        response.healthy = True
        response.version = "2.0.0"
        response.uptime_seconds = int(time.time() - self.start_time)
        return response

def serve():
    port = os.getenv("GRPC_PORT", "50051")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ai_service_pb2_grpc.add_AIServiceServicer_to_server(AIServiceServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"🚀 gRPC Server listening on {port}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()