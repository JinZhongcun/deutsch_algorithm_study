#!/usr/bin/env python3
"""
Phase 2: QSVMが優位性を示す「スイートスポット」を探す実験
複雑な非線形データでの量子カーネルの表現力を検証
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.algorithms import QSVC
import time
import warnings
warnings.filterwarnings('ignore')

class QSVMSweetSpotAnalysis:
    """QSVMの優位性を探る包括的な分析"""
    
    def __init__(self):
        self.results = {}
        
    def generate_datasets(self, n_samples=200, noise=0.1, random_state=42):
        """様々な複雑度のデータセットを生成"""
        datasets = {}
        
        # 1. 線形分離可能（CSVMが有利）
        X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, 
                         cluster_std=1.0, random_state=random_state)
        datasets['linear_separable'] = (X, y, 'Linear Separable')
        
        # 2. 月型データ（非線形、中程度の複雑さ）
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        datasets['moons'] = (X, y, 'Moons Dataset')
        
        # 3. 円形データ（非線形、高い複雑さ）
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.3, 
                           random_state=random_state)
        datasets['circles'] = (X, y, 'Circles Dataset')
        
        # 4. 高次元での複雑なパターン
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                  n_informative=2, n_clusters_per_class=2,
                                  random_state=random_state)
        datasets['complex_classification'] = (X, y, 'Complex Classification')
        
        return datasets
    
    def compare_kernels_on_dataset(self, X, y, dataset_name):
        """単一データセットでの各種カーネルの比較"""
        # データの準備
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {'dataset_name': dataset_name}
        
        # 1. Classical SVMs
        classical_kernels = ['linear', 'rbf', 'poly']
        for kernel in classical_kernels:
            start_time = time.perf_counter()
            
            if kernel == 'poly':
                clf = SVC(kernel=kernel, degree=3, random_state=42)
            else:
                clf = SVC(kernel=kernel, random_state=42)
                
            clf.fit(X_train_scaled, y_train)
            train_time = time.perf_counter() - start_time
            
            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[f'csvm_{kernel}'] = {
                'model': clf,
                'accuracy': accuracy,
                'train_time': train_time
            }
        
        # 2. Quantum SVMs with different feature maps
        quantum_configs = [
            ('ZZ_linear', ZZFeatureMap(2, reps=2, entanglement='linear')),
            ('ZZ_full', ZZFeatureMap(2, reps=2, entanglement='full')),
            ('Pauli', PauliFeatureMap(2, reps=2, paulis=['Z', 'ZZ']))
        ]
        
        for config_name, feature_map in quantum_configs:
            try:
                start_time = time.perf_counter()
                
                quantum_kernel = FidelityStatevectorKernel(feature_map=feature_map)
                qsvm = QSVC(quantum_kernel=quantum_kernel)
                qsvm.fit(X_train_scaled, y_train)
                
                train_time = time.perf_counter() - start_time
                
                y_pred = qsvm.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[f'qsvm_{config_name}'] = {
                    'model': qsvm,
                    'accuracy': accuracy,
                    'train_time': train_time,
                    'kernel_matrix': quantum_kernel.evaluate(X_train_scaled)
                }
            except Exception as e:
                print(f"QSVM {config_name} failed on {dataset_name}: {e}")
                results[f'qsvm_{config_name}'] = {
                    'accuracy': 0.0,
                    'train_time': float('inf'),
                    'error': str(e)
                }
        
        return results, X_train_scaled, X_test_scaled, y_train, y_test
    
    def visualize_decision_boundaries(self, X_train, y_train, results, dataset_name):
        """決定境界と量子カーネル行列の可視化"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Decision Boundaries and Quantum Kernels: {dataset_name}', 
                    fontsize=16, fontweight='bold')
        
        # メッシュグリッドの作成
        h = 0.02
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # 1. 古典SVMの決定境界
        classical_models = ['csvm_linear', 'csvm_rbf', 'csvm_poly']
        classical_titles = ['Linear SVM', 'RBF SVM', 'Polynomial SVM']
        
        for i, (model_key, title) in enumerate(zip(classical_models, classical_titles)):
            ax = fig.add_subplot(gs[0, i])
            
            if model_key in results and 'error' not in results[model_key]:
                model = results[model_key]['model']
                accuracy = results[model_key]['accuracy']
                
                # 決定境界
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
                ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
                
                # データ点
                scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                                   cmap='RdBu', edgecolors='black', s=50)
                
                ax.set_title(f'{title}\nAcc: {accuracy:.3f}')
            else:
                ax.text(0.5, 0.5, 'Failed', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{title}\nFailed')
            
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        
        # 2. 量子SVMの決定境界
        quantum_models = ['qsvm_ZZ_linear', 'qsvm_ZZ_full', 'qsvm_Pauli']
        quantum_titles = ['QSVM (ZZ Linear)', 'QSVM (ZZ Full)', 'QSVM (Pauli)']
        
        for i, (model_key, title) in enumerate(zip(quantum_models, quantum_titles)):
            ax = fig.add_subplot(gs[1, i])
            
            if model_key in results and 'error' not in results[model_key]:
                model = results[model_key]['model']
                accuracy = results[model_key]['accuracy']
                
                # データ点のみプロット（決定境界は複雑で時間がかかるため）
                scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                                   cmap='RdBu', edgecolors='black', s=50)
                
                ax.set_title(f'{title}\nAcc: {accuracy:.3f}')
            else:
                ax.text(0.5, 0.5, 'Failed', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{title}\nFailed')
            
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        
        # 3. 量子カーネル行列の比較
        for i, model_key in enumerate(quantum_models):
            ax = fig.add_subplot(gs[2, i])
            
            if model_key in results and 'kernel_matrix' in results[model_key]:
                kernel_matrix = results[model_key]['kernel_matrix']
                im = ax.imshow(kernel_matrix, cmap='viridis', aspect='auto')
                ax.set_title(f'Quantum Kernel Matrix\n({quantum_titles[i]})')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Sample Index')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'No Matrix', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('No Matrix Available')
        
        # 4. 性能比較
        ax = fig.add_subplot(gs[0:2, 3])
        
        model_names = []
        accuracies = []
        colors = []
        
        for model_key in classical_models + quantum_models:
            if model_key in results and 'error' not in results[model_key]:
                model_names.append(model_key.replace('csvm_', 'C-').replace('qsvm_', 'Q-'))
                accuracies.append(results[model_key]['accuracy'])
                colors.append('steelblue' if 'csvm' in model_key else 'coral')
        
        if model_names:
            bars = ax.barh(model_names, accuracies, color=colors)
            ax.set_xlabel('Accuracy')
            ax.set_title('Model Performance Comparison')
            ax.set_xlim(0, 1)
            
            # 値をバーの上に表示
            for bar, acc in zip(bars, accuracies):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{acc:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(f'qsvm_analysis_{dataset_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_entanglement_effect(self):
        """エンタングルメントの効果を分析"""
        print("\nAnalyzing entanglement effects...")
        
        # 複雑なデータセットを使用
        X, y = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # エンタングルメントの有無で比較
        entanglement_types = ['linear', 'full', 'circular']
        results = {}
        
        for entanglement in entanglement_types:
            try:
                feature_map = ZZFeatureMap(2, reps=2, entanglement=entanglement)
                quantum_kernel = FidelityStatevectorKernel(feature_map=feature_map)
                
                start_time = time.perf_counter()
                qsvm = QSVC(quantum_kernel=quantum_kernel)
                qsvm.fit(X_train_scaled, y_train)
                train_time = time.perf_counter() - start_time
                
                y_pred = qsvm.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # カーネル行列の特性を分析
                kernel_matrix = quantum_kernel.evaluate(X_train_scaled)
                kernel_mean = np.mean(kernel_matrix)
                kernel_std = np.std(kernel_matrix)
                
                results[entanglement] = {
                    'accuracy': accuracy,
                    'train_time': train_time,
                    'kernel_mean': kernel_mean,
                    'kernel_std': kernel_std,
                    'kernel_matrix': kernel_matrix
                }
                
                print(f"  {entanglement}: Acc={accuracy:.3f}, Time={train_time:.3f}s")
                
            except Exception as e:
                print(f"  {entanglement}: Failed - {e}")
                results[entanglement] = {'error': str(e)}
        
        # エンタングルメント効果の可視化
        self.visualize_entanglement_effects(results)
        
        return results
    
    def visualize_entanglement_effects(self, entanglement_results):
        """エンタングルメント効果の可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Effect of Quantum Entanglement on QSVM Performance', 
                    fontsize=16, fontweight='bold')
        
        entanglement_types = ['linear', 'full', 'circular']
        
        # 1. 精度比較
        ax = axes[0, 0]
        accuracies = []
        colors = ['blue', 'red', 'green']
        
        for ent_type in entanglement_types:
            if ent_type in entanglement_results and 'error' not in entanglement_results[ent_type]:
                accuracies.append(entanglement_results[ent_type]['accuracy'])
            else:
                accuracies.append(0)
        
        bars = ax.bar(entanglement_types, accuracies, color=colors, alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Entanglement Type')
        ax.set_ylim(0, 1)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. カーネル行列の比較
        for i, ent_type in enumerate(entanglement_types):
            ax = axes[1, i]
            
            if ent_type in entanglement_results and 'kernel_matrix' in entanglement_results[ent_type]:
                kernel_matrix = entanglement_results[ent_type]['kernel_matrix']
                im = ax.imshow(kernel_matrix, cmap='viridis', aspect='auto')
                ax.set_title(f'Kernel Matrix ({ent_type})')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'Failed', ha='center', va='center')
                ax.set_title(f'Kernel Matrix ({ent_type})')
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Sample Index')
        
        # 3. カーネル統計
        ax = axes[0, 1]
        kernel_means = []
        kernel_stds = []
        
        for ent_type in entanglement_types:
            if ent_type in entanglement_results and 'kernel_mean' in entanglement_results[ent_type]:
                kernel_means.append(entanglement_results[ent_type]['kernel_mean'])
                kernel_stds.append(entanglement_results[ent_type]['kernel_std'])
            else:
                kernel_means.append(0)
                kernel_stds.append(0)
        
        x = np.arange(len(entanglement_types))
        width = 0.35
        
        ax.bar(x - width/2, kernel_means, width, label='Mean', color='blue', alpha=0.7)
        ax.bar(x + width/2, kernel_stds, width, label='Std Dev', color='red', alpha=0.7)
        
        ax.set_ylabel('Value')
        ax.set_title('Kernel Matrix Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(entanglement_types)
        ax.legend()
        
        # 4. 実行時間比較
        ax = axes[0, 2]
        train_times = []
        
        for ent_type in entanglement_types:
            if ent_type in entanglement_results and 'train_time' in entanglement_results[ent_type]:
                train_times.append(entanglement_results[ent_type]['train_time'])
            else:
                train_times.append(0)
        
        bars = ax.bar(entanglement_types, train_times, color=colors, alpha=0.7)
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Training Time vs Entanglement')
        
        for bar, time_val in zip(bars, train_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                   f'{time_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('entanglement_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_comprehensive_analysis(self):
        """包括的な分析を実行"""
        print("Phase 2: Finding QSVM's Sweet Spot")
        print("="*50)
        
        # 1. 様々なデータセットの生成
        datasets = self.generate_datasets()
        
        # 2. 各データセットでの比較
        all_results = {}
        
        for dataset_key, (X, y, dataset_name) in datasets.items():
            print(f"\nAnalyzing dataset: {dataset_name}")
            
            results, X_train, X_test, y_train, y_test = self.compare_kernels_on_dataset(X, y, dataset_name)
            all_results[dataset_key] = results
            
            # 可視化
            self.visualize_decision_boundaries(X_train, y_train, results, dataset_name)
            print(f"  Saved: qsvm_analysis_{dataset_key}.png")
        
        # 3. エンタングルメント効果の分析
        entanglement_results = self.analyze_entanglement_effect()
        
        # 4. 結果のサマリー
        self.print_analysis_summary(all_results)
        
        return all_results, entanglement_results
    
    def print_analysis_summary(self, all_results):
        """分析結果のサマリーを表示"""
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*60)
        
        best_quantum_results = {}
        
        for dataset_key, results in all_results.items():
            dataset_name = results['dataset_name']
            print(f"\nDataset: {dataset_name}")
            print("-" * 40)
            
            # 最高性能の古典SVMを特定
            best_classical_acc = 0
            best_classical_model = ""
            
            # 最高性能の量子SVMを特定
            best_quantum_acc = 0
            best_quantum_model = ""
            
            for model_key, model_result in results.items():
                if model_key == 'dataset_name':
                    continue
                    
                if 'error' in model_result:
                    continue
                    
                acc = model_result['accuracy']
                
                if model_key.startswith('csvm_') and acc > best_classical_acc:
                    best_classical_acc = acc
                    best_classical_model = model_key
                elif model_key.startswith('qsvm_') and acc > best_quantum_acc:
                    best_quantum_acc = acc
                    best_quantum_model = model_key
            
            print(f"Best Classical: {best_classical_model} ({best_classical_acc:.3f})")
            print(f"Best Quantum:   {best_quantum_model} ({best_quantum_acc:.3f})")
            
            if best_quantum_acc > best_classical_acc:
                print(f"🎉 QUANTUM ADVANTAGE: +{best_quantum_acc - best_classical_acc:.3f}")
                best_quantum_results[dataset_name] = {
                    'advantage': best_quantum_acc - best_classical_acc,
                    'quantum_model': best_quantum_model,
                    'classical_model': best_classical_model
                }
            else:
                print(f"❌ Classical still better: -{best_classical_acc - best_quantum_acc:.3f}")
        
        # 量子優位性が見られた場合
        if best_quantum_results:
            print("\n🎯 QUANTUM SWEET SPOTS FOUND:")
            for dataset, result in best_quantum_results.items():
                print(f"  {dataset}: {result['quantum_model']} beats {result['classical_model']} "
                      f"by {result['advantage']:.3f}")
        else:
            print("\n❌ No clear quantum advantage found in current experiments")
            print("   This highlights the challenge of demonstrating NISQ-era quantum ML advantages")


if __name__ == "__main__":
    analyzer = QSVMSweetSpotAnalysis()
    all_results, entanglement_results = analyzer.run_comprehensive_analysis()