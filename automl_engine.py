import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import openml
import os
import json

class AutoMLEngine:
    def __init__(self, time_budget=30, metric='accuracy', task='classification'):
        self.time_budget = time_budget
        self.metric = metric
        self.task = task
        self.automl = AutoML()
        self.best_model = None
        self.best_config = None
        
    def train_from_openml(self, dataset_id: int):
        """
        從 OpenML 下載資料集並進行訓練
        例如: Titanic (ID: 31), Iris (ID: 61)
        """
        print(f"📥 Downloading dataset ID {dataset_id} from OpenML...")
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        
        # 簡單處理：將類別特徵轉為數值 (FLAML 其實可以自動處理，但這裡確保萬無一失)
        # 這裡直接交給 FLAML 處理
        
        return self.train(X, y)

    def train_from_csv(self, file_path: str, target_column: str):
        """
        從 CSV 檔案讀取資料並訓練
        """
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
            
        df = pd.read_csv(file_path)
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found in CSV."}
            
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return self.train(X, y)

    def train(self, X, y):
        """
        執行 AutoML 訓練流程
        """
        # 切分資料集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"🚀 Starting AutoML training (Budget: {self.time_budget}s)...")
        settings = {
            "time_budget": self.time_budget,  # total running time in seconds
            "metric": self.metric, 
            "task": self.task,
            "log_file_name": "automl.log",
        }
        
        self.automl.fit(X_train=X_train, y_train=y_train, **settings)
        
        # 預測與評估
        print("📊 Predicting on test set...")
        y_pred = self.automl.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.best_model = self.automl.model
        self.best_config = self.automl.best_config
        
        # 整理特徵重要性 (如果模型支援)
        feature_importance = {}
        try:
            if hasattr(self.automl.model.estimator, "feature_importances_"):
                fi = self.automl.model.estimator.feature_importances_
                feature_importance = dict(zip(X.columns, fi))
                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
        except:
            feature_importance = {"info": "Model does not support feature importance extraction directly."}

        result = {
            "best_estimator": self.automl.best_estimator,
            "best_loss": self.automl.best_loss,
            "test_accuracy": acc,
            "training_duration": self.automl.best_config_train_time,
            "feature_importance": feature_importance,
            "classification_report": str(report) # 轉字串避免太長
        }
        
        return result

if __name__ == "__main__":
    # Local Test
    engine = AutoMLEngine(time_budget=10)
    # Test with Iris (OpenML ID: 61)
    result = engine.train_from_openml(61)
    print(json.dumps(result, indent=2, default=str))
