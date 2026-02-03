"""
MLflow Experiment Tracking Configuration
Author: Erdinc Erdogan
Purpose: Configures MLflow for ML experiment tracking and model versioning with graceful degradation when MLflow is unavailable.
References:
- MLflow Tracking API
- ML Experiment Management
- Model Registry Patterns
Usage:
    setup_mlflow(experiment_name="btc_price_prediction")
    log_model_metrics(model, metrics={"accuracy": 0.87, "loss": 0.13})
"""

import os
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ============================================
# GRACEFUL MLFLOW IMPORT
# ============================================

MLFLOW_AVAILABLE = False

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
    logger.info("✅ MLflow loaded successfully")
except ImportError:
    mlflow = None
    MlflowClient = None
    logger.warning("⚠️ MLflow not installed. Experiment tracking disabled.")


# ============================================
# CONFIGURATION
# ============================================

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
DEFAULT_EXPERIMENT_NAME = 'neuraltrade_trading_signals'


# ============================================
# MOCK CLASSES FOR GRACEFUL DEGRADATION
# ============================================

class MockRun:
    """Mock MLflow run when MLflow is not available."""
    
    class Info:
        run_id = "mock_run_id"
    
    info = Info()


@contextmanager
def mock_start_run(*args, **kwargs):
    """Mock context manager for MLflow runs."""
    yield MockRun()


# ============================================
# SETUP FUNCTIONS
# ============================================

def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Initialize MLflow tracking for an experiment.
    Returns None if MLflow is not available.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning(f"⚠️ MLflow not available. Skipping experiment setup: {experiment_name}")
        return None
    
    uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                tags=tags or {'project': 'neuraltrade', 'type': 'trading'}
            )
            logger.info(f"✅ Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"📊 Using existing experiment: {experiment_name} (ID: {experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        logger.error(f"❌ MLflow setup failed: {e}")
        return None


def start_run(
    run_name: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, str]] = None,
):
    """Start an MLflow run. Returns mock context if MLflow unavailable."""
    if not MLFLOW_AVAILABLE:
        return mock_start_run(run_name=run_name, nested=nested, tags=tags)
    return mlflow.start_run(run_name=run_name, nested=nested, tags=tags)


# ============================================
# LOGGING FUNCTIONS
# ============================================

def log_params(params: Dict[str, Any]) -> None:
    """Log multiple parameters."""
    if not MLFLOW_AVAILABLE:
        logger.debug(f"[Mock] Params: {params}")
        return
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log multiple metrics."""
    if not MLFLOW_AVAILABLE:
        logger.debug(f"[Mock] Metrics: {metrics}")
        return
    mlflow.log_metrics(metrics, step=step)


def log_model_metrics(
    model: Any,
    model_name: str,
    metrics: Dict[str, float],
    params: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Log a trained model with its metrics to MLflow.
    Returns None if MLflow is not available.
    """
    if not MLFLOW_AVAILABLE:
        logger.info(f"[Mock] Model logged: {model_name}, metrics: {metrics}")
        return "mock_run_id"
    
    with mlflow.start_run() as run:
        if params:
            mlflow.log_params(params)
        
        mlflow.log_metrics(metrics)
        
        try:
            if hasattr(model, 'save'):
                mlflow.tensorflow.log_model(model, model_name)
            elif hasattr(model, 'save_pretrained'):
                mlflow.pyfunc.log_model(model_name, python_model=model)
            else:
                mlflow.sklearn.log_model(model, model_name)
        except Exception as e:
            logger.warning(f"Could not log model: {e}")
            mlflow.log_dict({'model_info': str(type(model))}, 'model_info.json')
        
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, name)
        
        logger.info(f"✅ Logged model {model_name} (Run ID: {run.info.run_id})")
        return run.info.run_id


def log_signal_prediction(
    symbol: str,
    predicted_action: str,
    confidence: float,
    actual_action: Optional[str] = None,
    model_name: str = 'signal_model',
) -> None:
    """Log a signal prediction for tracking prediction accuracy."""
    if not MLFLOW_AVAILABLE:
        logger.debug(f"[Mock] Signal: {symbol} -> {predicted_action} ({confidence:.2%})")
        return
    
    with mlflow.start_run(run_name=f"signal_{symbol}", nested=True):
        mlflow.log_params({
            'symbol': symbol,
            'model': model_name,
            'predicted_action': predicted_action,
        })
        mlflow.log_metrics({'confidence': confidence})
        
        if actual_action:
            mlflow.log_params({'actual_action': actual_action})
            mlflow.log_metrics({
                'correct': 1.0 if predicted_action == actual_action else 0.0
            })


# ============================================
# MODEL REGISTRY
# ============================================

def register_model(
    model_uri: str,
    name: str,
    tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Register a model in the MLflow Model Registry."""
    if not MLFLOW_AVAILABLE:
        logger.warning(f"[Mock] Model registration skipped: {name}")
        return "mock_version_1"
    
    client = MlflowClient()
    
    try:
        client.create_registered_model(name, tags=tags)
    except Exception:
        pass
    
    result = client.create_model_version(source=model_uri, name=name)
    logger.info(f"✅ Registered model {name} v{result.version}")
    return result.version


def get_latest_model_version(name: str, stage: str = "Production") -> Optional[str]:
    """Get the latest model version in a stage."""
    if not MLFLOW_AVAILABLE:
        return None
    
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(name, stages=[stage])
        if versions:
            return versions[0].version
    except Exception as e:
        logger.error(f"Error getting model version: {e}")
    return None


def load_production_model(name: str):
    """Load the production version of a registered model."""
    if not MLFLOW_AVAILABLE:
        raise RuntimeError("MLflow not available. Cannot load model.")
    
    model_uri = f"models:/{name}/Production"
    return mlflow.pyfunc.load_model(model_uri)


# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_experiment_runs(
    experiment_name: str,
    max_results: int = 100,
) -> List[Dict[str, Any]]:
    """Get all runs for an experiment."""
    if not MLFLOW_AVAILABLE:
        return []
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        return []
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
    )
    return runs.to_dict('records')


def cleanup_old_runs(
    experiment_name: str,
    keep_latest: int = 50,
) -> int:
    """Delete old runs from an experiment."""
    if not MLFLOW_AVAILABLE:
        return 0
    
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if not experiment:
        return 0
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    
    deleted = 0
    for i, run in enumerate(runs.itertuples()):
        if i >= keep_latest:
            client.delete_run(run.run_id)
            deleted += 1
    
    logger.info(f"🗑️ Deleted {deleted} old runs from {experiment_name}")
    return deleted


def is_mlflow_available() -> bool:
    """Check if MLflow is available."""
    return MLFLOW_AVAILABLE


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print(f"MLflow Available: {is_mlflow_available()}")
    
    setup_mlflow(experiment_name="test_experiment")
    
    with start_run(run_name="test_run"):
        log_params({
            'learning_rate': 0.01,
            'epochs': 100,
            'batch_size': 32,
        })
        log_metrics({
            'accuracy': 0.87,
            'loss': 0.13,
            'f1_score': 0.85,
        })
        print("✅ Test run completed (mock or real)")