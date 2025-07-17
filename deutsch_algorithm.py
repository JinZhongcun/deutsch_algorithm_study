import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

class DeutschAlgorithm:
    """
    Deutschアルゴリズムの実装
    1ビット関数f(x)が定数関数かバランス関数かを1回のクエリで判定
    """
    
    def __init__(self):
        self.simulator = AerSimulator()
        self.circuits = {}
        self.results = {}
    
    def create_oracle(self, oracle_type):
        """
        オラクル回路を作成
        
        Parameters:
        oracle_type (str): 'constant_0', 'constant_1', 'balanced_id', 'balanced_not'
        
        Returns:
        QuantumCircuit: オラクル回路
        """
        oracle = QuantumCircuit(2, name=f"Oracle({oracle_type})")
        
        if oracle_type == 'constant_0':
            # f(x) = 0 for all x
            pass  # 何もしない（恒等演算）
        elif oracle_type == 'constant_1':
            # f(x) = 1 for all x
            oracle.x(1)  # 補助量子ビットにXゲートを適用
        elif oracle_type == 'balanced_id':
            # f(x) = x
            oracle.cx(0, 1)  # CNOTゲート
        elif oracle_type == 'balanced_not':
            # f(x) = NOT x
            oracle.cx(0, 1)  # CNOTゲート
            oracle.x(1)      # 補助量子ビットにXゲートを適用
        else:
            raise ValueError(f"Unknown oracle type: {oracle_type}")
        
        return oracle
    
    def create_deutsch_circuit(self, oracle_type):
        """
        完全なDeutschアルゴリズムの回路を作成
        
        Parameters:
        oracle_type (str): オラクルのタイプ
        
        Returns:
        QuantumCircuit: Deutsch回路
        """
        qc = QuantumCircuit(2, 1)
        qc.name = f"Deutsch({oracle_type})"
        
        # Step 1: 初期化
        qc.x(1)  # 補助量子ビットを|1⟩に設定
        
        # Step 2: アダマールゲートで重ね合わせ状態を作成
        qc.h(0)
        qc.h(1)
        
        # Step 3: オラクルを適用
        oracle = self.create_oracle(oracle_type)
        qc.append(oracle, [0, 1])
        
        # Step 4: 最初の量子ビットにアダマールゲートを適用
        qc.h(0)
        
        # Step 5: 最初の量子ビットを測定
        qc.measure(0, 0)
        
        return qc
    
    def run_deutsch(self, oracle_type, shots=1024):
        """
        Deutschアルゴリズムを実行
        
        Parameters:
        oracle_type (str): オラクルのタイプ
        shots (int): 測定回数
        
        Returns:
        dict: 測定結果
        """
        # 回路を作成
        circuit = self.create_deutsch_circuit(oracle_type)
        self.circuits[oracle_type] = circuit
        
        # トランスパイルして実行
        compiled_circuit = transpile(circuit, self.simulator)
        job = self.simulator.run(compiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # 結果を保存
        self.results[oracle_type] = {
            'counts': counts,
            'is_constant': '0' in counts and counts.get('0', 0) > shots * 0.9,
            'is_balanced': '1' in counts and counts.get('1', 0) > shots * 0.9
        }
        
        return self.results[oracle_type]
    
    def run_all_cases(self, shots=1024):
        """
        全てのオラクルタイプでDeutschアルゴリズムを実行
        """
        oracle_types = ['constant_0', 'constant_1', 'balanced_id', 'balanced_not']
        
        for oracle_type in oracle_types:
            print(f"\n実行中: {oracle_type}")
            result = self.run_deutsch(oracle_type, shots)
            
            # 結果の解釈
            if result['is_constant']:
                print(f"  → 判定: 定数関数 (測定結果は主に'0')")
            elif result['is_balanced']:
                print(f"  → 判定: バランス関数 (測定結果は主に'1')")
            else:
                print(f"  → エラー: 明確な判定ができません")
            
            print(f"  測定結果: {result['counts']}")
    
    def visualize_circuits(self):
        """
        全ての回路を可視化
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (oracle_type, circuit) in enumerate(self.circuits.items()):
            ax = axes[i]
            circuit.draw(output='mpl', ax=ax)
            ax.set_title(f"Deutsch Circuit: {oracle_type}")
        
        plt.tight_layout()
        plt.savefig('deutsch_circuits.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_results(self):
        """
        測定結果を可視化
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (oracle_type, result) in enumerate(self.results.items()):
            ax = axes[i]
            counts = result['counts']
            
            # ヒストグラムを作成
            bars = ax.bar(counts.keys(), counts.values(), color='steelblue')
            ax.set_xlabel('Measurement Result')
            ax.set_ylabel('Count')
            ax.set_title(f'{oracle_type}\n{"Constant Function" if result["is_constant"] else "Balanced Function"}')
            ax.set_ylim(0, max(counts.values()) * 1.1)
            
            # 値をバーの上に表示
            for bar, count in zip(bars, counts.values()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('deutsch_results.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """
    Deutschアルゴリズムのデモを実行
    """
    print("=== Deutschアルゴリズムの実装 ===")
    print("1ビット関数が定数関数かバランス関数かを1回のクエリで判定します。")
    
    # Deutschアルゴリズムのインスタンスを作成
    deutsch = DeutschAlgorithm()
    
    # 全てのケースを実行
    deutsch.run_all_cases(shots=1024)
    
    # 回路と結果を可視化
    print("\n回路図を生成中...")
    deutsch.visualize_circuits()
    print("回路図を 'deutsch_circuits.png' に保存しました")
    
    print("\n結果のヒストグラムを生成中...")
    deutsch.visualize_results()
    print("結果を 'deutsch_results.png' に保存しました")
    
    # サマリー
    print("\n=== 結果のサマリー ===")
    print("定数関数:")
    print("  - constant_0 (f(x) = 0): 測定結果は'0'")
    print("  - constant_1 (f(x) = 1): 測定結果は'0'")
    print("バランス関数:")
    print("  - balanced_id (f(x) = x): 測定結果は'1'")
    print("  - balanced_not (f(x) = NOT x): 測定結果は'1'")


if __name__ == "__main__":
    main()