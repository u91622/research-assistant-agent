import pytest
from automl_v3_final import AutoMLEngine
from tools import train_tabular_model

def test_iris_training():
    engine = AutoMLEngine(time_budget=5)
    result = engine.train_from_openml(61)
    
    assert "best_estimator" in result
    assert "test_accuracy" in result
    assert result["test_accuracy"] > 0.5
    
def test_tool_execution():
    result = train_tabular_model.invoke({"dataset_source": "openml:61"})
    
    assert "AutoML Training Complete" in result
    assert "Accuracy" in result
    assert "Precision" in result
    assert "Confusion Matrix" in result
    assert "Best Estimator" in result
    
    # Iris 是一個簡單的資料集，準確率應該要很高
    # 但因為時間只有 5-10 秒，不做過度嚴格要求
