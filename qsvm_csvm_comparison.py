import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
# from qiskit.primitives import Sampler  # FidelityQuantumKernelで自動的に使用される


class QuantumClassicalSVMComparison:
    """
    QSVMとCSVMの性能比較を行うクラス
    """
    
    def __init__(self, digits_pair=(3, 4), n_components=2, random_state=42):
        """
        Parameters:
        digits_pair (tuple): 比較する数字のペア
        n_components (int): PCAで削減する次元数
        random_state (int): 乱数シード
        """
        self.digits_pair = digits_pair
        self.n_components = n_components
        self.random_state = random_state
        self.results = {}
        
        # データの準備
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # モデル
        self.csvm_models = {}
        self.qsvm_model = None
        
    def prepare_data(self):
        """
        データセットの準備とPCAによる次元削減
        """
        print(f"\nデータ準備中: 数字 {self.digits_pair[0]} と {self.digits_pair[1]}")
        
        # データセットをロード
        digits = load_digits()
        
        # 指定された2つの数字のデータのみを抽出
        mask = np.isin(digits.target, self.digits_pair)
        X = digits.data[mask]
        y = digits.target[mask]
        
        # ラベルを0と1に変換
        y = (y == self.digits_pair[1]).astype(int)
        
        print(f"元のデータ形状: {X.shape}")
        print(f"クラス分布: {np.bincount(y)}")
        
        # 訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        # データの正規化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # PCAで次元削減
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.X_train = pca.fit_transform(X_train_scaled)
        self.X_test = pca.transform(X_test_scaled)
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"PCA後のデータ形状: {self.X_train.shape}")
        print(f"説明分散比: {pca.explained_variance_ratio_}")
        print(f"累積説明分散比: {np.sum(pca.explained_variance_ratio_):.3f}")
        
        return pca
    
    def train_classical_svm(self, kernel_types=['linear', 'rbf', 'poly', 'sigmoid']):
        """
        複数のカーネルでClassical SVMを訓練
        """
        print("\n=== Classical SVM の訓練 ===")
        
        for kernel in kernel_types:
            print(f"\nカーネル: {kernel}")
            
            # 実行時間の測定
            start_time = time.perf_counter()
            
            # モデルの作成と訓練
            if kernel == 'poly':
                model = SVC(kernel=kernel, degree=3, random_state=self.random_state)
            else:
                model = SVC(kernel=kernel, random_state=self.random_state)
            
            model.fit(self.X_train, self.y_train)
            
            train_time = time.perf_counter() - start_time
            
            # 予測と評価
            start_time = time.perf_counter()
            y_pred = model.predict(self.X_test)
            predict_time = time.perf_counter() - start_time
            
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # 結果を保存
            self.csvm_models[kernel] = model
            self.results[f'csvm_{kernel}'] = {
                'model': model,
                'train_time': train_time,
                'predict_time': predict_time,
                'accuracy': accuracy,
                'y_pred': y_pred,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            print(f"  訓練時間: {train_time:.4f}秒")
            print(f"  予測時間: {predict_time:.4f}秒")
            print(f"  精度: {accuracy:.3f}")
    
    def train_quantum_svm(self, feature_map_reps=2):
        """
        Quantum SVMを訓練
        """
        print("\n=== Quantum SVM の訓練 ===")
        
        # 量子特徴マップの作成
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_components,
            reps=feature_map_reps,
            entanglement='full'
        )
        
        print(f"特徴マップ: ZZFeatureMap (reps={feature_map_reps})")
        
        # 量子カーネルの作成
        # FidelityQuantumKernelはデフォルトでSamplerを使用
        quantum_kernel = FidelityQuantumKernel(
            feature_map=feature_map,
            enforce_psd=True,
            evaluate_duplicates='off_diagonal'
        )
        
        # QSVCの訓練
        start_time = time.perf_counter()
        
        self.qsvm_model = QSVC(quantum_kernel=quantum_kernel)
        self.qsvm_model.fit(self.X_train, self.y_train)
        
        train_time = time.perf_counter() - start_time
        
        # 予測と評価
        start_time = time.perf_counter()
        y_pred = self.qsvm_model.predict(self.X_test)
        predict_time = time.perf_counter() - start_time
        
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # 結果を保存
        self.results['qsvm'] = {
            'model': self.qsvm_model,
            'train_time': train_time,
            'predict_time': predict_time,
            'accuracy': accuracy,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'feature_map': feature_map
        }
        
        print(f"  訓練時間: {train_time:.4f}秒")
        print(f"  予測時間: {predict_time:.4f}秒")
        print(f"  精度: {accuracy:.3f}")
    
    def visualize_decision_boundaries(self):
        """
        決定境界の可視化
        """
        # グリッドポイントの作成
        h = .02  # メッシュのステップサイズ
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # プロット設定
        models_to_plot = ['linear', 'rbf', 'poly', 'qsvm']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, model_type in enumerate(models_to_plot):
            ax = axes[idx]
            
            if model_type == 'qsvm':
                model = self.qsvm_model
                title = 'Quantum SVM'
            else:
                model = self.csvm_models[model_type]
                title = f'Classical SVM ({model_type})'
            
            # 決定境界の予測
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # コントアプロット
            ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu)
            
            # 訓練データ点のプロット
            scatter = ax.scatter(self.X_train[:, 0], self.X_train[:, 1], 
                               c=self.y_train, cmap=plt.cm.RdBu, 
                               edgecolors='black', s=50)
            
            # テストデータ点のプロット（別マーカー）
            ax.scatter(self.X_test[:, 0], self.X_test[:, 1], 
                      c=self.y_test, cmap=plt.cm.RdBu, 
                      marker='^', edgecolors='black', s=50, alpha=0.6)
            
            ax.set_xlabel('第1主成分')
            ax.set_ylabel('第2主成分')
            ax.set_title(f'{title}\n精度: {self.results.get(f"csvm_{model_type}" if model_type != "qsvm" else "qsvm", {}).get("accuracy", 0):.3f}')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
        
        plt.tight_layout()
        plt.savefig(f'decision_boundaries_{self.digits_pair[0]}_{self.digits_pair[1]}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_performance(self):
        """
        性能比較の可視化
        """
        # データ準備
        models = list(self.results.keys())
        train_times = [self.results[m]['train_time'] for m in models]
        predict_times = [self.results[m]['predict_time'] for m in models]
        accuracies = [self.results[m]['accuracy'] for m in models]
        
        # プロット
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 訓練時間の比較
        bars1 = ax1.bar(models, train_times, color=['steelblue']*4 + ['coral'])
        ax1.set_ylabel('時間 (秒)')
        ax1.set_title('訓練時間の比較')
        ax1.set_xticklabels(models, rotation=45)
        
        # バーの上に値を表示
        for bar, time in zip(bars1, train_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.3f}', ha='center', va='bottom')
        
        # 予測時間の比較
        bars2 = ax2.bar(models, predict_times, color=['steelblue']*4 + ['coral'])
        ax2.set_ylabel('時間 (秒)')
        ax2.set_title('予測時間の比較')
        ax2.set_xticklabels(models, rotation=45)
        
        for bar, time in zip(bars2, predict_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.4f}', ha='center', va='bottom')
        
        # 精度の比較
        bars3 = ax3.bar(models, accuracies, color=['steelblue']*4 + ['coral'])
        ax3.set_ylabel('精度')
        ax3.set_title('分類精度の比較')
        ax3.set_xticklabels(models, rotation=45)
        ax3.set_ylim(0, 1.1)
        
        for bar, acc in zip(bars3, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'performance_comparison_{self.digits_pair[0]}_{self.digits_pair[1]}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """
        結果のサマリーを表示
        """
        print("\n=== 性能比較サマリー ===")
        print(f"数字ペア: {self.digits_pair[0]} vs {self.digits_pair[1]}")
        print("-" * 60)
        print(f"{'モデル':<15} {'訓練時間(秒)':<12} {'予測時間(秒)':<12} {'精度':<8}")
        print("-" * 60)
        
        for model_name, result in self.results.items():
            print(f"{model_name:<15} {result['train_time']:<12.4f} "
                  f"{result['predict_time']:<12.4f} {result['accuracy']:<8.3f}")
        
        # QSVMとベストCSVMの比較
        csvm_accuracies = {k: v['accuracy'] for k, v in self.results.items() if k.startswith('csvm')}
        best_csvm = max(csvm_accuracies, key=csvm_accuracies.get)
        qsvm_acc = self.results['qsvm']['accuracy']
        best_csvm_acc = csvm_accuracies[best_csvm]
        
        print("\n=== QSVMの評価 ===")
        print(f"最高性能のCSVM: {best_csvm} (精度: {best_csvm_acc:.3f})")
        print(f"QSVM精度: {qsvm_acc:.3f}")
        print(f"精度差: {qsvm_acc - best_csvm_acc:+.3f}")
        
        # 実行時間の比較
        qsvm_train_time = self.results['qsvm']['train_time']
        best_csvm_train_time = self.results[best_csvm]['train_time']
        print(f"\n訓練時間比: QSVM/CSVM = {qsvm_train_time/best_csvm_train_time:.1f}倍")


def run_experiment(digits_pairs=[(3, 4), (1, 2)]):
    """
    複数の数字ペアで実験を実行
    """
    all_results = {}
    
    for digits_pair in digits_pairs:
        print(f"\n{'='*60}")
        print(f"実験開始: 数字 {digits_pair[0]} vs {digits_pair[1]}")
        print(f"{'='*60}")
        
        # 比較実験の実行
        comparison = QuantumClassicalSVMComparison(digits_pair=digits_pair)
        
        # データ準備
        pca = comparison.prepare_data()
        
        # Classical SVMの訓練
        comparison.train_classical_svm()
        
        # Quantum SVMの訓練
        comparison.train_quantum_svm()
        
        # 結果の可視化
        print("\n決定境界を可視化中...")
        comparison.visualize_decision_boundaries()
        
        print("\n性能比較グラフを作成中...")
        comparison.compare_performance()
        
        # サマリー表示
        comparison.print_summary()
        
        # 結果を保存
        all_results[digits_pair] = comparison.results
    
    return all_results


if __name__ == "__main__":
    # 実験を実行
    results = run_experiment(digits_pairs=[(3, 4), (1, 2)])
    
    print("\n\n=== 実験完了 ===")
    print("生成されたファイル:")
    print("- decision_boundaries_3_4.png: 数字3,4の決定境界")
    print("- performance_comparison_3_4.png: 数字3,4の性能比較")
    print("- decision_boundaries_1_2.png: 数字1,2の決定境界")
    print("- performance_comparison_1_2.png: 数字1,2の性能比較")