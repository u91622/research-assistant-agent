import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import openml
import os
import time

class AutoMLEngine:
    def __init__(self, time_budget=30, metric='accuracy', task='classification'):
        self.time_budget = time_budget
        self.metric = metric
        self.task = task
        self.best_model = None
        self.best_score = -1
        self.best_name = ""
        
    def train_from_openml(self, dataset_id: int):
        """從 OpenML 下載資料集並進行訓練"""
        print(f"📥 Downloading dataset ID {dataset_id} from OpenML...")
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute, dataset_format="dataframe"
            )
            return self.train(X, y)
        except Exception as e:
            return {"error": f"OpenML Download Failed: {str(e)}"}

    def train_from_csv(self, file_path: str, target_column: str):
        """從 CSV 檔案讀取資料並訓練"""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        try:    
            df = pd.read_csv(file_path)
            if target_column not in df.columns:
                return {"error": f"Target column '{target_column}' not found."}
                
            X = df.drop(columns=[target_column])
            y = df[target_column]
            return self.train(X, y)
        except Exception as e:
            return {"error": f"CSV Read Failed: {str(e)}"}

    def train(self, X, y):
        """
        執行輕量級 AutoML 訓練流程 (Native Sklearn)
        """
        # 手動處理缺值與類別特徵
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # 定義候選模型
        models = [
            ("RandomForest", RandomForestClassifier(n_jobs=-1, random_state=42)),
            ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
            ("LogisticRegression", LogisticRegression(max_iter=1000))
        ]

        # Encoding Target Variable if needed (e.g. for strings '0', '1' or 'died', 'survived')
        le = LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"🚀 Starting AutoML training (Budget: {self.time_budget}s)...")
        start_time = time.time()
        
        results = []
        
        for name, model in models:
            if time.time() - start_time > self.time_budget:
                break
                
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            
            try:
                pipe.fit(X_train, y_train)
                score = pipe.score(X_test, y_test)
                results.append((name, score, pipe))
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = pipe
                    self.best_name = name
            except Exception as e:
                print(f"Model {name} failed: {e}")

        if not self.best_model:
            return {"error": "Training failed for all models."}

        # 整理結果
        y_pred = self.best_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 計算額外指標
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        # 自動判斷 average 参数 (二元 vs 多類別)
        avg_method = 'binary' if len(np.unique(y)) == 2 else 'weighted'
        
        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # 嘗試提取特徵重要性 (對於 Tree-based model)
        feature_importance = {}
        try:
            regressor = self.best_model.named_steps['classifier']
            preprocessor = self.best_model.named_steps['preprocessor']
            
            if hasattr(regressor, 'feature_importances_'):
                importances = regressor.feature_importances_
                
                # 取得轉換後的特徵名稱
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except:
                    # Fallback if get_feature_names_out fails
                    feature_names = numeric_features.tolist() + [f"cat_{i}" for i in range(len(importances) - len(numeric_features))]
                
                # 確保長度一致
                if len(feature_names) == len(importances):
                    feature_importance = dict(zip(feature_names, importances))
                    # 去除前綴 (例如 'num__', 'cat__') 讓名稱更乾淨
                    clean_importance = {}
                    for k, v in feature_importance.items():
                        clean_k = k.replace("num__", "").replace("cat__", "")
                        clean_importance[clean_k] = v
                    
                    feature_importance = dict(sorted(clean_importance.items(), key=lambda item: item[1], reverse=True))
        except Exception as e:
            print(f"Feature importance extraction failed: {e}")
            pass

        # 模型保存 (Deployment Ready)
        import joblib
        model_filename = f"best_model_{int(time.time())}.pkl"
        joblib.dump(self.best_model, model_filename)

        return {
            "best_estimator": self.best_name,
            "test_accuracy": self.best_score,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "training_duration": time.time() - start_time,
            "feature_importance": feature_importance,
            "classification_report": str(report),
            "saved_model_path": model_filename
        }
