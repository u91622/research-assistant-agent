from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from duckduckgo_search import DDGS
import warnings

# 忽略 DuckDuckGoSearch 的更名警告與可能的資源警告
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
warnings.filterwarnings("ignore", category=ResourceWarning)

@tool
def multiply(a: int, b: int) -> int:
    """相乘兩個整數。"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """相加兩個整數。"""
    return a + b

@tool
def search_duckduckgo(query: str) -> str:
    """使用 DuckDuckGo 搜尋網路。"""
    with DDGS() as ddgs:
        # 減少搜尋結果數量以降低 Token 消耗 (避免 Groq 免費版 Rate Limit)
        results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        return "\n\n".join([f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}" for r in results])

@tool
def train_tabular_model(dataset_source: str, target_column: str = "target") -> str:
    """
    使用 AutoML 自動訓練機器學習模型。
    
    Args:
        dataset_source (str): 
            可以是 OpenML 的資料集 ID (例如 "openml:61" 代表 Iris, "openml:31" 代表 Titanic)，
            或者是本地 CSV 檔案路徑 (例如 "data.csv")。
        target_column (str): 
            目標欄位名稱 (Target Column)，預設為 "class" 或 "target"。
            OpenML 資料集通常不需要此參數。

    Returns:
        str: 訓練結果摘要，包含準確率、最佳模型與特徵重要性。
    """
    try:
        import importlib
        import automl_v3_final
        importlib.reload(automl_v3_final)
        from automl_v3_final import AutoMLEngine
        # 為了 Demo 快速回應，設定預算為 10-30 秒
        engine = AutoMLEngine(time_budget=20) 
        
        result = {}
        # Smart Shortcut for Titanic
        if dataset_source.lower() == "titanic":
            dataset_source = "titanic.csv"
            if target_column == "target": # Only override if default
                target_column = "Survived"
                
        if dataset_source.startswith("openml:"):
            ds_id = int(dataset_source.split(":")[1])
            result = engine.train_from_openml(ds_id)
        else:
            result = engine.train_from_csv(dataset_source, target_column)
            
        if "error" in result:
            return f"Error: {result['error']}"
            
        # 建立易讀的摘要
        summary = (
            f"✅ AutoML Training Complete!\n"
            f"- Best Estimator: {result['best_estimator']}\n"
            f"- Accuracy: {result['test_accuracy']:.4f}\n"
            f"- Precision: {result.get('precision', 0):.4f}\n"
            f"- Recall: {result.get('recall', 0):.4f}\n"
            f"- F1 Score: {result.get('f1_score', 0):.4f}\n"
            f"- Confusion Matrix: {result.get('confusion_matrix')}\n"
            f"- Training Time: {result['training_duration']:.2f}s\n"
            f"- 💾 Model Saved: {result.get('saved_model_path')} (Ready for Deployment)\n"
        )
        
        if result.get('feature_importance'):
            top_features = list(result['feature_importance'].keys())[:3]
            summary += f"- Top Features: {top_features}\n"
            
        return summary
    except Exception as e:
        return f"AutoML Failed: {str(e)}"

tools = [multiply, add, search_duckduckgo, train_tabular_model]
