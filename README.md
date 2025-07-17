# Quantum Computing Study: Deutsch Algorithm & QSVM Analysis

*量子コンピューティング学習：Deutschアルゴリズムと量子SVM分析*

## Overview / 概要

This project implements the Deutsch algorithm and evaluates Quantum Support Vector Machine (QSVM) performance compared to classical SVMs. The study demonstrates both the power and limitations of current quantum computing approaches.

本プロジェクトはDeutschアルゴリズムの実装と量子サポートベクターマシン（QSVM）の性能評価を行い、現在の量子コンピューティング手法の力と限界を実証します。

## Key Findings / 主要な発見

### ✅ Deutsch Algorithm: Clear Quantum Advantage
**Classical**: Requires 2 oracle queries  
**Quantum**: Requires only 1 oracle query  
**Advantage**: 50% reduction in query complexity through quantum parallelism

### ❌ QSVM: Current Limitations
- **Performance**: Classical SVM consistently outperforms QSVM (13-38% accuracy gap)
- **Speed**: QSVM is 20-50x slower than classical methods
- **Reality Check**: NISQ-era constraints prevent theoretical advantages from materializing

## Contents / 内容

### 1. Deutsch Algorithm Implementation
- Quantum parallelism demonstration using function property determination
- Quantum circuit simulation with Qiskit
- State vector evolution visualization

### 2. QSVM vs CSVM Comprehensive Comparison
- Performance evaluation on handwritten digit recognition
- Multiple datasets: digits 1vs2, 3vs4, moons, circles
- Detailed analysis of quantum kernel effectiveness

### 3. Essential Differences Analysis
- Root cause analysis of quantum vs classical computational approaches
- Identification of problem-algorithm compatibility factors
- Educational insights into quantum computing reality vs expectations

## File Structure / ファイル構成

### Core Implementation
- `deutsch_algorithm.py`: Deutsch algorithm implementation
- `quantum_classical_comparison.py`: Phase 1 experiments (query complexity)
- `qsvm_sweet_spot.py`: Phase 2 experiments (QSVM analysis)
- `final_optimized_experiment.py`: Optimized QSVM experiments

### Analysis & Reports
- `ESSENTIAL_DIFFERENCES_ANALYSIS.md`: Comprehensive analysis of quantum vs classical differences
- `FINAL_REPORT.md`: Complete experimental report
- Various visualization outputs (PNG format)

## Environment Setup / 環境構築

```bash
# Using Docker
docker build -t quantum-assignment .
docker run -it quantum-assignment

# Or install dependencies directly
pip install -r requirements.txt
```

## Running Experiments / 実験実行

```bash
# Phase 1: Deutsch Algorithm Analysis
python quantum_classical_comparison.py

# Phase 2: QSVM Sweet Spot Analysis  
python qsvm_sweet_spot.py

# Optimized QSVM Experiments
python final_optimized_experiment.py
```

## Dependencies / 必要なライブラリ

- qiskit==0.45.0
- qiskit-machine-learning==0.7.0
- scikit-learn==1.3.2
- matplotlib==3.8.2
- numpy==1.24.3

## Key Insights / 重要な洞察

### 🎯 Quantum Computing Reality
**Myth**: "Quantum computers are universally faster"  
**Reality**: "Quantum computers excel at specific, well-structured problems"

### 🔬 The Power of Interference
Quantum advantage comes from:
1. **Superposition**: Parallel evaluation of multiple possibilities
2. **Interference**: Canceling unwanted solutions  
3. **Amplification**: Enhancing desired outcomes

### 📊 Current Limitations
- **NISQ Constraints**: Noise and limited qubit count
- **Algorithm Maturity**: Classical methods are highly optimized
- **Problem Compatibility**: Not all problems benefit from quantum approaches

## Educational Value / 教育的価値

This study provides hands-on experience with:
- **Quantum Advantage**: Where and why it occurs (Deutsch algorithm)
- **Current Reality**: Practical limitations of NISQ-era devices
- **Critical Thinking**: Evaluating quantum computing claims objectively
- **Research Skills**: Systematic experimental design and analysis

## Future Directions / 今後の方向性

1. **Fault-Tolerant Era**: Await error-corrected quantum computers
2. **Algorithm Design**: Develop quantum-native problem formulations  
3. **Hybrid Approaches**: Combine quantum and classical strengths
4. **Application Areas**: Focus on problems with proven quantum advantages

---

*This repository serves as an educational resource for understanding both the promise and current limitations of quantum computing technology.*

*本リポジトリは量子コンピューティング技術の可能性と現在の限界の両方を理解するための教育リソースとして機能します。*