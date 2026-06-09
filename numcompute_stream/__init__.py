"""NumCompute-Stream: streaming tree-based learning with NumPy."""
from .tree import DecisionTreeClassifier
from .ensemble import EnsembleClassifier, RandomForestClassifier
from .preprocessing import StandardScaler, Imputer, OneHotEncoder
from .pipeline import Pipeline
from .stream import StreamTrainer
from .metrics import StreamingMetric, StreamingConfusionMatrix, StreamingAUC, accuracy_score
from .stats import StreamingStats

__all__ = [
    "DecisionTreeClassifier", "EnsembleClassifier", "RandomForestClassifier",
    "StandardScaler", "Imputer", "OneHotEncoder", "Pipeline", "StreamTrainer",
    "StreamingMetric", "StreamingConfusionMatrix", "StreamingAUC", "accuracy_score", "StreamingStats"
]
