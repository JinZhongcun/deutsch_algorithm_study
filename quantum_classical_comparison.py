#!/usr/bin/env python3
"""
量子vs古典の本質的な違いを明確にする実験
Phase 1: Deutschアルゴリズムのクエリ複雑度比較
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_qsphere, plot_bloch_multivector
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

class DeutschComparison:
    """Deutschアルゴリズムの量子vs古典比較"""
    
    def __init__(self):
        self.oracle_calls_classical = 0
        self.oracle_calls_quantum = 0
        self.simulator = AerSimulator()
        
    def classical_oracle(self, x, oracle_type):
        """古典的なオラクル関数"""
        self.oracle_calls_classical += 1
        
        if oracle_type == 'constant_0':
            return 0
        elif oracle_type == 'constant_1':
            return 1
        elif oracle_type == 'balanced_id':
            return x
        elif oracle_type == 'balanced_not':
            return 1 - x
            
    def classical_deutsch(self, oracle_type):
        """古典的なDeutschアルゴリズム"""
        self.oracle_calls_classical = 0
        
        # 必ず2回のクエリが必要
        f0 = self.classical_oracle(0, oracle_type)
        f1 = self.classical_oracle(1, oracle_type)
        
        # 定数関数かバランス関数かを判定
        is_constant = (f0 == f1)
        
        return {
            'is_constant': is_constant,
            'oracle_calls': self.oracle_calls_classical,
            'f(0)': f0,
            'f(1)': f1
        }
    
    def quantum_oracle(self, circuit, oracle_type):
        """量子オラクル"""
        self.oracle_calls_quantum += 1
        
        if oracle_type == 'constant_0':
            # 何もしない
            pass
        elif oracle_type == 'constant_1':
            circuit.x(1)
        elif oracle_type == 'balanced_id':
            circuit.cx(0, 1)
        elif oracle_type == 'balanced_not':
            circuit.cx(0, 1)
            circuit.x(1)
            
    def quantum_deutsch_with_states(self, oracle_type):
        """量子Deutschアルゴリズム（状態の追跡付き）"""
        self.oracle_calls_quantum = 0
        states = {}
        
        # 初期回路
        qc = QuantumCircuit(2, 1)
        
        # Step 1: 初期化
        qc.x(1)
        states['after_init'] = Statevector.from_instruction(qc)
        
        # Step 2: Hadamard gates
        qc.h(0)
        qc.h(1)
        states['after_hadamard'] = Statevector.from_instruction(qc)
        
        # Step 3: Oracle
        self.quantum_oracle(qc, oracle_type)
        states['after_oracle'] = Statevector.from_instruction(qc)
        
        # Step 4: Final Hadamard
        qc.h(0)
        states['before_measurement'] = Statevector.from_instruction(qc)
        
        # Step 5: Measurement
        qc.measure(0, 0)
        
        # Execute
        job = self.simulator.run(transpile(qc, self.simulator), shots=1)
        result = job.result()
        counts = result.get_counts()
        
        measurement = int(list(counts.keys())[0])
        is_constant = (measurement == 0)
        
        return {
            'is_constant': is_constant,
            'oracle_calls': self.oracle_calls_quantum,
            'circuit': qc,
            'states': states,
            'measurement': measurement
        }
    
    def visualize_query_complexity(self):
        """クエリ複雑度の比較を可視化"""
        oracle_types = ['constant_0', 'constant_1', 'balanced_id', 'balanced_not']
        
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Deutsch Algorithm: Quantum vs Classical Query Complexity', 
                    fontsize=16, fontweight='bold')
        
        # 1. Query count comparison
        ax1 = fig.add_subplot(gs[0, :])
        
        classical_calls = []
        quantum_calls = []
        
        for oracle in oracle_types:
            c_result = self.classical_deutsch(oracle)
            q_result = self.quantum_deutsch_with_states(oracle)
            classical_calls.append(c_result['oracle_calls'])
            quantum_calls.append(q_result['oracle_calls'])
        
        x = np.arange(len(oracle_types))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, classical_calls, width, label='Classical', color='blue')
        bars2 = ax1.bar(x + width/2, quantum_calls, width, label='Quantum', color='red')
        
        ax1.set_xlabel('Oracle Type')
        ax1.set_ylabel('Number of Oracle Calls')
        ax1.set_title('Oracle Query Complexity: Classical Always Needs 2, Quantum Only 1')
        ax1.set_xticks(x)
        ax1.set_xticklabels(oracle_types)
        ax1.legend()
        ax1.set_ylim(0, 3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Classical algorithm flow
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.text(0.5, 0.9, 'Classical Algorithm Flow', 
                ha='center', fontsize=14, fontweight='bold')
        
        flow_text = """
        1. Query f(0) → Get value
        2. Query f(1) → Get value
        3. Compare f(0) and f(1)
        4. If equal → Constant
           If different → Balanced
        
        MUST query function TWICE
        No way to reduce this!
        """
        ax2.text(0.1, 0.4, flow_text, fontsize=11, va='center')
        ax2.axis('off')
        
        # 3. Quantum algorithm flow
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.text(0.5, 0.9, 'Quantum Algorithm Flow', 
                ha='center', fontsize=14, fontweight='bold')
        
        flow_text = """
        1. Create superposition |0⟩+|1⟩
        2. Query f(superposition) ONCE
        3. Use interference
        4. Measure → Get answer
        
        Quantum parallelism allows
        evaluating f(0) AND f(1)
        simultaneously!
        """
        ax3.text(0.1, 0.4, flow_text, fontsize=11, va='center')
        ax3.axis('off')
        
        # 4. Quantum advantage summary
        ax4 = fig.add_subplot(gs[2, :])
        ax4.text(0.5, 0.8, 'Key Insight: Quantum Parallelism', 
                ha='center', fontsize=14, fontweight='bold')
        
        summary_text = """
        The quantum advantage is NOT about speed of computation, but about the NUMBER of queries needed.
        
        • Classical: Must check f(0) and f(1) separately → 2 queries minimum
        • Quantum: Can check both simultaneously using superposition → 1 query is sufficient
        
        This is a provable separation between quantum and classical computation!
        For n-bit Deutsch-Jozsa: Classical needs 2^(n-1)+1 queries, Quantum needs only 1.
        """
        ax4.text(0.5, 0.3, summary_text, ha='center', fontsize=11, 
                wrap=True, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('deutsch_query_complexity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_quantum_states(self, oracle_type='balanced_id'):
        """量子状態の変化を可視化"""
        result = self.quantum_deutsch_with_states(oracle_type)
        states = result['states']
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'Quantum State Evolution in Deutsch Algorithm ({oracle_type})', 
                    fontsize=16, fontweight='bold')
        
        # 各ステップの状態を表示
        steps = ['after_init', 'after_hadamard', 'after_oracle', 'before_measurement']
        step_names = ['After Initialization', 'After Hadamard', 'After Oracle', 'Before Measurement']
        
        for i, (step, name) in enumerate(zip(steps, step_names)):
            ax = plt.subplot(2, 4, i+1)
            state = states[step]
            
            # 状態ベクトルの振幅を表示
            amplitudes = state.data
            basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
            
            # 実部と虚部を分けて表示
            real_parts = np.real(amplitudes)
            imag_parts = np.imag(amplitudes)
            
            x = np.arange(len(basis_labels))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, real_parts, width, label='Real', color='blue', alpha=0.7)
            bars2 = ax.bar(x + width/2, imag_parts, width, label='Imaginary', color='red', alpha=0.7)
            
            ax.set_xlabel('Basis State')
            ax.set_ylabel('Amplitude')
            ax.set_title(name)
            ax.set_xticks(x)
            ax.set_xticklabels(basis_labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1, 1)
            
            # 確率も表示（下のサブプロット）
            ax2 = plt.subplot(2, 4, i+5)
            probabilities = np.abs(amplitudes)**2
            bars = ax2.bar(basis_labels, probabilities, color='green', alpha=0.7)
            ax2.set_xlabel('Basis State')
            ax2.set_ylabel('Probability')
            ax2.set_title(f'Measurement Probabilities at {name}')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            
            # 値をバーの上に表示
            for bar, prob in zip(bars, probabilities):
                if prob > 0.01:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{prob:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'deutsch_state_evolution_{oracle_type}.png', dpi=300, bbox_inches='tight')
        plt.close()


def run_phase1_experiments():
    """Phase 1: Deutschアルゴリズムの本質的な違いを示す実験"""
    print("Phase 1: Deutsch Algorithm - Quantum vs Classical Comparison")
    print("="*60)
    
    comparison = DeutschComparison()
    
    # 1. クエリ複雑度の比較
    print("\n1. Creating query complexity comparison...")
    comparison.visualize_query_complexity()
    print("   Saved: deutsch_query_complexity.png")
    
    # 2. 量子状態の進化を可視化（各オラクルタイプ）
    print("\n2. Visualizing quantum state evolution...")
    for oracle_type in ['constant_0', 'balanced_id']:
        comparison.visualize_quantum_states(oracle_type)
        print(f"   Saved: deutsch_state_evolution_{oracle_type}.png")
    
    # 3. 定量的な比較結果
    print("\n3. Quantitative Results:")
    print("-"*40)
    
    oracle_types = ['constant_0', 'constant_1', 'balanced_id', 'balanced_not']
    
    for oracle in oracle_types:
        c_result = comparison.classical_deutsch(oracle)
        q_result = comparison.quantum_deutsch_with_states(oracle)
        
        print(f"\nOracle: {oracle}")
        print(f"  Classical: {c_result['oracle_calls']} queries, f(0)={c_result['f(0)']}, f(1)={c_result['f(1)']}, constant={c_result['is_constant']}")
        print(f"  Quantum:   {q_result['oracle_calls']} query, measurement={q_result['measurement']}, constant={q_result['is_constant']}")
    
    print("\n" + "="*60)
    print("Key Takeaway: Quantum algorithm achieves PROVABLE advantage")
    print("by reducing oracle queries from 2 to 1 using superposition!")


if __name__ == "__main__":
    run_phase1_experiments()