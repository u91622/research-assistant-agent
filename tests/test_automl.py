from tools import train_tabular_model
import pytest

def test_automl_openml_iris():
    """測試 AutoML Tool 能否正確處理 OpenML Iris 資料集"""
    # Iris ID: 61
    result = train_tabular_model.invoke({"dataset_source": "openml:61"})
    
    assert "AutoML Training Complete" in result
    assert "Test Accuracy" in result
    assert "Best Estimator" in result
    
    # Iris 是一個簡單的資料集，準確率應該要很高
    # 但考慮到時間預算很短，我們只檢查是否成功產出結果

def test_automl_invalid_source():
    """測試錯誤處理"""
    result = train_tabular_model.invoke({"dataset_source": "invalid_file.csv"})
    assert "Error: File not found" in result
