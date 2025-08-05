#!/usr/bin/env python3
"""
SVM Decision Boundary Comparison: Classical vs Quantum-inspired
査読者の指摘に対応するため、決定境界を明確に区別して可視化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SVM Decision Boundary Comparison")
print("Classical vs Quantum-inspired SVM")
print("="*60)

# データ読み込み
print("\n1. Loading YouTube dataset...")
df = pd.read_csv('youtube_top_new_complete.csv')
print(f"   Loaded {len(df)} samples")

# 2クラス分類問題の設定
print("\n2. Creating binary classification problem...")
median_views = df['views'].median()
df['high_views'] = (df['views'] > median_views).astype(int)
print(f"   Median views: {median_views:,.0f}")
print(f"   High views: {sum(df['high_views'])} samples")
print(f"   Low views: {len(df) - sum(df['high_views'])} samples")

# 特徴量選択（重要度の高い2つ）
print("\n3. Selecting features...")
features = ['subscribers', 'video_duration', 'brightness', 'colorfulness']
X = df[features].fillna(0)
y = df['high_views']

# データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCAで2次元に削減（可視化のため）
print("   Applying PCA for 2D visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"   Explained variance ratio: {pca.explained_variance_ratio_}")

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y
)

print("\n4. Training models...")

# Classical SVM
print("   Training Classical SVM (RBF kernel)...")
classical_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
classical_svm.fit(X_train, y_train)
classical_pred = classical_svm.predict(X_test)
classical_acc = accuracy_score(y_test, classical_pred)
print(f"   Classical SVM Accuracy: {classical_acc:.3f}")

# Quantum-inspired SVM
print("   Training Quantum-inspired SVM...")
# Nystroemで高次元特徴空間へマッピング（量子特徴マップを模倣）
quantum_svm = Pipeline([
    ('feature_map', Nystroem(kernel='rbf', gamma=0.5, n_components=100, random_state=42)),
    ('svm', SVC(kernel='linear', C=1.0, random_state=42))
])
quantum_svm.fit(X_train, y_train)
quantum_pred = quantum_svm.predict(X_test)
quantum_acc = accuracy_score(y_test, quantum_pred)
print(f"   Quantum-inspired SVM Accuracy: {quantum_acc:.3f}")

# 決定境界の可視化
print("\n5. Plotting decision boundaries...")

# メッシュグリッドの作成
h = 0.02  # メッシュのステップサイズ
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 図の作成
plt.figure(figsize=(12, 8))

# Classical SVMの決定領域と境界
Z_classical = classical_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z_classical = Z_classical.reshape(xx.shape)

# Quantum-inspired SVMの決定領域と境界
Z_quantum = quantum_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z_quantum = Z_quantum.reshape(xx.shape)

# 背景の決定領域を薄く表示（Classical）
plt.contourf(xx, yy, Z_classical, alpha=0.2, cmap=plt.cm.Blues, levels=[-0.5, 0.5, 1.5])

# 決定境界の描画
# Classical SVM - 青色の実線
classical_contour = plt.contour(xx, yy, Z_classical, colors='blue', 
                                linewidths=2.5, linestyles='solid', levels=[0.5])

# Quantum-inspired SVM - 赤色の破線
quantum_contour = plt.contour(xx, yy, Z_quantum, colors='red', 
                              linewidths=2.5, linestyles='dashed', levels=[0.5])

# データポイントのプロット
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                      cmap=plt.cm.coolwarm, edgecolors='black', 
                      alpha=0.6, s=30)

# 凡例の作成（重要！）
legend_elements = [
    plt.Line2D([0], [0], color='blue', lw=2.5, linestyle='-', 
               label=f'Classical SVM (Acc: {classical_acc:.3f})'),
    plt.Line2D([0], [0], color='red', lw=2.5, linestyle='--', 
               label=f'Quantum-inspired SVM (Acc: {quantum_acc:.3f})')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

# グラフの装飾
plt.xlabel('First Principal Component', fontsize=12)
plt.ylabel('Second Principal Component', fontsize=12)
plt.title('Decision Boundary Comparison: Classical vs Quantum-inspired SVM\n' + 
          'YouTube Video View Count Classification (High/Low Views)', 
          fontsize=14, pad=20)

# カラーバーの追加
cbar = plt.colorbar(scatter, ticks=[0, 1])
cbar.ax.set_yticklabels(['Low Views', 'High Views'])

# グリッドの追加
plt.grid(True, alpha=0.3)

# 保存
output_file = 'svm_decision_boundary_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n6. Saved decision boundary comparison to: {output_file}")

# サブプロット版も作成（別々に表示）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Classical SVM（左）
ax1.contourf(xx, yy, Z_classical, alpha=0.4, cmap=plt.cm.Blues)
ax1.contour(xx, yy, Z_classical, colors='blue', linewidths=2.5, levels=[0.5])
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, 
            edgecolors='black', alpha=0.6, s=30)
ax1.set_title(f'Classical SVM (RBF kernel)\nAccuracy: {classical_acc:.3f}', fontsize=14)
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')
ax1.grid(True, alpha=0.3)

# Quantum-inspired SVM（右）
ax2.contourf(xx, yy, Z_quantum, alpha=0.4, cmap=plt.cm.Reds)
ax2.contour(xx, yy, Z_quantum, colors='red', linewidths=2.5, levels=[0.5])
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, 
                       edgecolors='black', alpha=0.6, s=30)
ax2.set_title(f'Quantum-inspired SVM (Nystroem + Linear)\nAccuracy: {quantum_acc:.3f}', fontsize=14)
ax2.set_xlabel('First Principal Component')
ax2.set_ylabel('Second Principal Component')
ax2.grid(True, alpha=0.3)

# 共通のカラーバー
cbar2 = fig.colorbar(scatter2, ax=ax2, ticks=[0, 1])
cbar2.ax.set_yticklabels(['Low Views', 'High Views'])

plt.suptitle('Decision Boundary Comparison: Side by Side', fontsize=16)
plt.tight_layout()

# サブプロット版も保存
subplot_file = 'svm_decision_boundary_subplot.png'
plt.savefig(subplot_file, dpi=300, bbox_inches='tight')
print(f"   Also saved subplot version to: {subplot_file}")

# 性能比較サマリー
print("\n" + "="*60)
print("Performance Summary:")
print(f"  Classical SVM:        {classical_acc:.3f}")
print(f"  Quantum-inspired SVM: {quantum_acc:.3f}")
print(f"  Difference:           {quantum_acc - classical_acc:+.3f}")

# 決定境界の違いを定量化
# 境界付近のサンプルで予測が異なる割合を計算
mesh_points = np.c_[xx.ravel(), yy.ravel()]
classical_mesh_pred = classical_svm.predict(mesh_points)
quantum_mesh_pred = quantum_svm.predict(mesh_points)
disagreement_rate = np.mean(classical_mesh_pred != quantum_mesh_pred)
print(f"  Boundary disagreement: {disagreement_rate:.1%}")

if disagreement_rate < 0.05:
    print("\n⚠️  Note: The decision boundaries are very similar (< 5% disagreement).")
    print("  This suggests both methods converge to similar solutions for this dataset.")
else:
    print(f"\n✓ The decision boundaries show clear differences ({disagreement_rate:.1%} disagreement).")
    print("  The visualization effectively distinguishes between the two methods.")

print("="*60)
print("Analysis complete!")
print("="*60)