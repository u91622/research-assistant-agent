import matplotlib.pyplot as plt
import numpy as np

def plot_benchmark_results():
    """
    請將此函式複製到您的 Google Colab Notebook 中執行以生成圖表。
    此圖表視覺化了 vLLM 與 TGI 在不同模型下的吞吐量比較。
    """
    
    # 範例數據 (請替換為您實際跑出的數據)
    frameworks = ['vLLM', 'HF TGI']
    throughput = [1950, 1420]  # tokens/sec
    
    # 設定圖表樣式
    plt.figure(figsize=(10, 6))
    bars = plt.bar(frameworks, throughput, color=['#4CAF50', '#FF5722'])
    
    # 加入標籤
    plt.title('Inference Throughput Comparison (Llama-3.2-1B)', fontsize=15)
    plt.ylabel('Throughput (tokens/s)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在柱狀圖上方顯示數值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
                
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("請將此檔案內容或 plot_benchmark_results 函式複製到 Colab 中執行以查看圖表。")
    plot_benchmark_results()
