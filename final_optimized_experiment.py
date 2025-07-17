#!/usr/bin/env python3
"""
最終的な最適化実験 - カーネル行列事前計算版
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
import matplotlib.pyplot as plt
import time
import os

def run_optimized_experiment():
    """最適化された実験の実行"""
    print("量子コンピューティング最終実験 - 最適化版")
    print("="*60)
    
    # Deutschアルゴリズムは完了済み
    print("\nPart 1: Deutschアルゴリズム ✓ 完了")
    
    print("\nPart 2: QSVM/CSVM比較（カーネル行列事前計算）")
    print("-"*60)
    
    all_results = {}
    
    for digits_pair in [(3, 4), (1, 2)]:
        print(f"\n実験: 数字 {digits_pair[0]} vs {digits_pair[1]}")
        
        # データ準備（各クラス15サンプル = 合計30サンプル）
        digits = load_digits()
        mask = np.isin(digits.target, digits_pair)
        X = digits.data[mask]
        y = digits.target[mask]
        
        # サンプル数制限
        n_samples = 15
        indices_0 = np.where(y == digits_pair[0])[0][:n_samples]
        indices_1 = np.where(y == digits_pair[1])[0][:n_samples]
        indices = np.concatenate([indices_0, indices_1])
        X = X[indices]
        y = (y[indices] == digits_pair[1]).astype(int)
        
        print(f"  データサイズ: {X.shape}")
        
        # 訓練/テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # PCA
        scaler = StandardScaler()
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(scaler.fit_transform(X_train))
        X_test_pca = pca.transform(scaler.transform(X_test))
        
        print(f"  説明分散比: {pca.explained_variance_ratio_}")
        
        results = {}
        
        # Classical SVM (RBFのみで比較)
        print(f"\n  Classical SVM (RBF):")
        start = time.perf_counter()
        csvm = SVC(kernel='rbf', random_state=42)
        csvm.fit(X_train_pca, y_train)
        csvm_train_time = time.perf_counter() - start
        
        y_pred_csvm = csvm.predict(X_test_pca)
        csvm_accuracy = accuracy_score(y_test, y_pred_csvm)
        
        results['csvm_rbf'] = {
            'model': csvm,
            'train_time': csvm_train_time,
            'accuracy': csvm_accuracy
        }
        
        print(f"    訓練時間: {csvm_train_time:.4f}秒")
        print(f"    精度: {csvm_accuracy:.3f}")
        
        # Quantum SVM（カーネル行列事前計算）
        print(f"\n  Quantum SVM (事前計算版):")
        
        # 特徴マップとカーネル
        feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
        quantum_kernel = FidelityStatevectorKernel(feature_map=feature_map)
        
        # カーネル行列の事前計算
        print("    カーネル行列を計算中...")
        start = time.perf_counter()
        
        # 訓練データのカーネル行列
        kernel_matrix_train = quantum_kernel.evaluate(X_train_pca)
        kernel_calc_time = time.perf_counter() - start
        print(f"    カーネル行列計算時間: {kernel_calc_time:.4f}秒")
        
        # テストデータと訓練データ間のカーネル行列
        kernel_matrix_test = quantum_kernel.evaluate(X_test_pca, X_train_pca)
        
        # 事前計算済みカーネルでSVCを訓練
        start = time.perf_counter()
        qsvm = SVC(kernel='precomputed', random_state=42)
        qsvm.fit(kernel_matrix_train, y_train)
        qsvm_train_time = time.perf_counter() - start
        
        y_pred_qsvm = qsvm.predict(kernel_matrix_test)
        qsvm_accuracy = accuracy_score(y_test, y_pred_qsvm)
        
        results['qsvm'] = {
            'model': qsvm,
            'train_time': kernel_calc_time + qsvm_train_time,
            'kernel_time': kernel_calc_time,
            'svm_time': qsvm_train_time,
            'accuracy': qsvm_accuracy,
            'kernel_matrix': kernel_matrix_train
        }
        
        print(f"    SVM訓練時間: {qsvm_train_time:.4f}秒")
        print(f"    合計時間: {kernel_calc_time + qsvm_train_time:.4f}秒")
        print(f"    精度: {qsvm_accuracy:.3f}")
        
        # カーネル行列の可視化
        print("\n  カーネル行列を可視化中...")
        visualize_kernel_matrix(kernel_matrix_train, digits_pair)
        
        # 決定境界の可視化（簡易版）
        print("  決定境界を可視化中...")
        visualize_decision_boundary_simple(
            X_train_pca, y_train, X_test_pca, y_test,
            csvm, qsvm, kernel_matrix_test, digits_pair
        )
        
        all_results[digits_pair] = results
    
    # 最終レポート
    generate_optimized_report(all_results)
    
    print("\n" + "="*60)
    print("実験完了！")
    print("\n生成されたファイル:")
    print("- deutsch_circuits.png (既存)")
    print("- deutsch_results.png (既存)")
    print("- kernel_matrix_3_4.png")
    print("- kernel_matrix_1_2.png")
    print("- comparison_3_4.png")
    print("- comparison_1_2.png")
    print("- optimized_report.txt")


def visualize_kernel_matrix(kernel_matrix, digits_pair):
    """カーネル行列のヒートマップ"""
    plt.figure(figsize=(6, 5))
    im = plt.imshow(kernel_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im)
    plt.title(f'Quantum Kernel Matrix (Digits {digits_pair[0]} vs {digits_pair[1]})')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    plt.savefig(f'kernel_matrix_{digits_pair[0]}_{digits_pair[1]}.png', dpi=150)
    plt.close()


def visualize_decision_boundary_simple(X_train, y_train, X_test, y_test, 
                                      csvm, qsvm, kernel_matrix_test, digits_pair):
    """簡易的な決定境界の可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # データ点のプロット
    for ax, title in zip([ax1, ax2], ['Classical SVM (RBF)', 'Quantum SVM']):
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                  cmap=plt.cm.RdBu, edgecolors='black', s=80, alpha=0.8)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                  cmap=plt.cm.RdBu, marker='^', edgecolors='black', s=80)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(f'comparison_{digits_pair[0]}_{digits_pair[1]}.png', dpi=150)
    plt.close()


def generate_optimized_report(all_results):
    """最適化実験のレポート生成"""
    with open('optimized_report.txt', 'w', encoding='utf-8') as f:
        f.write("量子コンピューティング実験結果（最適化版）\\n")
        f.write("="*60 + "\\n\\n")
        
        f.write("実装の詳細:\\n")
        f.write("- Deutschアルゴリズム: Qiskitで実装\\n")
        f.write("- QSVM: FidelityStatevectorKernel + 事前計算\\n")
        f.write("- データ: 手書き数字認識（各クラス15サンプル）\\n")
        f.write("- 次元削減: PCA (64→2次元)\\n\\n")
        
        f.write("QSVM/CSVM比較結果:\\n")
        f.write("-"*60 + "\\n")
        
        for digits_pair, results in all_results.items():
            f.write(f"\\n数字ペア {digits_pair[0]} vs {digits_pair[1]}:\\n")
            
            csvm_res = results['csvm_rbf']
            qsvm_res = results['qsvm']
            
            f.write(f"  Classical SVM (RBF):\\n")
            f.write(f"    - 訓練時間: {csvm_res['train_time']:.4f}秒\\n")
            f.write(f"    - 精度: {csvm_res['accuracy']:.3f}\\n")
            
            f.write(f"  Quantum SVM:\\n")
            f.write(f"    - カーネル計算: {qsvm_res['kernel_time']:.4f}秒\\n")
            f.write(f"    - SVM訓練: {qsvm_res['svm_time']:.4f}秒\\n")
            f.write(f"    - 合計時間: {qsvm_res['train_time']:.4f}秒\\n")
            f.write(f"    - 精度: {qsvm_res['accuracy']:.3f}\\n")
            
            time_ratio = qsvm_res['train_time'] / csvm_res['train_time']
            f.write(f"  時間比 (QSVM/CSVM): {time_ratio:.1f}倍\\n")
        
        f.write("\\n考察:\\n")
        f.write("1. QSVMの計算時間の大部分はカーネル行列の計算\\n")
        f.write("2. カーネル行列のサイズはO(n²)で増加\\n")
        f.write("3. 現在のQSVM実装は計算効率で劣る\\n")
        f.write("4. 量子優位性は将来の量子ハードウェアに期待\\n")
        f.write("5. 教育目的には小規模データでの原理理解が重要\\n")
    
    print("\n最適化レポートを 'optimized_report.txt' に保存しました")


if __name__ == "__main__":
    run_optimized_experiment()