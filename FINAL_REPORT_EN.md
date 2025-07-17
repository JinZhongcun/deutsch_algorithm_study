# Quantum Computing Assignment 3: Final Report

**Date**: July 17, 2025  
**Content**: Deutsch Algorithm Implementation and QSVM/CSVM Performance Comparison

## Executive Summary

This report presents a comprehensive analysis of quantum computing through two distinct experiments: the Deutsch algorithm demonstration and Quantum Support Vector Machine (QSVM) performance evaluation. Our findings reveal both the remarkable theoretical advantages of quantum computing and the practical challenges facing current NISQ-era implementations.

## 1. Introduction

### 1.1 Objectives
- Implement and analyze the Deutsch algorithm to demonstrate quantum parallelism
- Compare QSVM performance against classical SVM across multiple datasets
- Identify the essential differences between quantum and classical computational approaches
- Evaluate the current state of quantum machine learning applications

### 1.2 Experimental Approach
The study was conducted in two phases:
- **Phase 1**: Deutsch Algorithm - demonstrating provable quantum advantage
- **Phase 2**: QSVM Analysis - exploring quantum machine learning reality

## 2. Methodology

### 2.1 Environment Setup
- **Platform**: Docker containerization for reproducibility
- **Primary Framework**: Qiskit 0.45.0 for quantum circuit simulation
- **ML Libraries**: scikit-learn 1.3.2, qiskit-machine-learning 0.7.0
- **Visualization**: matplotlib 3.8.2 for comprehensive data analysis

### 2.2 Dataset Selection
- **Primary Dataset**: sklearn.datasets.load_digits (handwritten digit recognition)
- **Target Classes**: Digits 3 vs 4, and 1 vs 2 for binary classification
- **Dimensionality Reduction**: PCA from 64 to 2 dimensions for visualization
- **Additional Datasets**: synthetic datasets (moons, circles) for comprehensive analysis

## 3. Results

### 3.1 Phase 1: Deutsch Algorithm

#### 3.1.1 Quantum vs Classical Query Complexity
| Oracle Type | Classical Queries | Quantum Queries | Advantage |
|-------------|------------------|-----------------|-----------|
| constant_0  | 2                | 1               | 50% reduction |
| constant_1  | 2                | 1               | 50% reduction |
| balanced_id | 2                | 1               | 50% reduction |
| balanced_not| 2                | 1               | 50% reduction |

**Key Finding**: The quantum algorithm achieves consistent 50% reduction in oracle queries across all function types, demonstrating provable computational advantage.

#### 3.1.2 Quantum State Evolution Analysis
State vector tracking revealed the mechanism of quantum advantage:
1. **Superposition Creation**: Hadamard gates create equal probability amplitudes
2. **Parallel Evaluation**: Oracle operates on superposed states simultaneously  
3. **Interference**: Final Hadamard gate creates constructive/destructive interference
4. **Measurement**: Interference patterns reveal function properties with single query

### 3.2 Phase 2: QSVM Performance Analysis

#### 3.2.1 Performance Comparison Results
| Dataset | Best Classical Accuracy | Best Quantum Accuracy | Performance Gap |
|---------|------------------------|----------------------|----------------|
| Digits 3 vs 4 | 1.000 (RBF SVM) | 0.867 (ZZ FeatureMap) | -13.3% |
| Digits 1 vs 2 | 1.000 (RBF SVM) | 0.783 (ZZ FeatureMap) | -21.7% |
| Moons Dataset | 0.983 (RBF SVM) | 0.600 (ZZ FeatureMap) | -38.3% |
| Circles Dataset | 1.000 (RBF SVM) | 0.783 (ZZ FeatureMap) | -21.7% |

#### 3.2.2 Computational Performance
- **Classical SVM**: ~0.001 seconds (highly optimized)
- **Quantum SVM**: ~0.025 seconds (20-25x slower)
- **Kernel Computation**: Dominant bottleneck in QSVM implementation

#### 3.2.3 Entanglement Effect Analysis
Testing different entanglement structures (linear, full, circular) showed no significant performance differences, suggesting that the current feature map designs do not effectively utilize quantum entanglement for these datasets.

## 4. Analysis and Discussion

### 4.1 Essential Differences Identified

#### 4.1.1 Computational Principles
**Deutsch Algorithm Success Factors**:
- Problem designed for quantum advantage (query complexity reduction)
- Interference-based information extraction
- Theoretical guarantees with mathematical proofs

**QSVM Challenge Factors**:
- Heuristic approach without theoretical guarantees
- NISQ-era limitations (noise, limited qubits)
- Feature map design challenges
- Highly optimized classical competition

#### 4.1.2 Problem-Algorithm Compatibility
The stark contrast between Phase 1 and Phase 2 results highlights a fundamental insight: **quantum advantage is highly problem-dependent**. The Deutsch algorithm represents a "quantum-native" problem where quantum properties directly translate to computational benefits, while QSVM attempts to apply quantum computing to problems that may not inherently benefit from quantum approaches.

### 4.2 Current Limitations Analysis

#### 4.2.1 NISQ-Era Constraints
1. **Noise Sensitivity**: Quantum states are fragile and easily disrupted
2. **Limited Qubit Count**: Restricts problem size and feature map complexity
3. **Gate Error Accumulation**: Affects kernel matrix accuracy
4. **Optimization Challenges**: Quantum parameter optimization is non-trivial

#### 4.2.2 Classical Algorithm Maturity
Classical SVMs benefit from decades of optimization:
- Highly efficient implementations
- Well-understood hyperparameter tuning
- Robust performance across diverse datasets
- Extensive theoretical foundations

### 4.3 Educational Insights

#### 4.3.1 Quantum Computing Reality Check
This study demonstrates the importance of:
- **Realistic Expectations**: Quantum computing is not universally superior
- **Problem Selection**: Quantum advantage requires appropriate problem structure
- **Current Limitations**: NISQ devices have significant practical constraints
- **Research Direction**: Focus on quantum-native algorithm design

#### 4.3.2 The Power of Interference
The Deutsch algorithm showcases quantum computing's true strength: the ability to use **interference** to extract information efficiently. This is fundamentally different from classical parallel processing and represents the core of quantum computational advantage.

## 5. Conclusions

### 5.1 Primary Findings

1. **Proven Quantum Advantage**: The Deutsch algorithm demonstrates clear, measurable quantum advantage through reduced query complexity.

2. **Current ML Limitations**: QSVM implementations currently cannot match classical SVM performance on standard datasets.

3. **Problem Dependency**: Quantum advantage is highly dependent on problem structure and algorithm design compatibility.

4. **Interference as Key**: Quantum interference, not just superposition, is crucial for computational advantage.

### 5.2 Implications for Quantum Computing

#### 5.2.1 Short-term (NISQ Era)
- Focus on problems with proven quantum advantage
- Develop noise-resilient algorithms
- Improve quantum error mitigation techniques
- Create quantum-classical hybrid approaches

#### 5.2.2 Long-term (Fault-Tolerant Era)
- Design quantum-native problem formulations
- Explore new application domains
- Develop quantum-specific optimization techniques
- Integrate quantum computing into broader computational workflows

### 5.3 Research Directions

1. **Algorithm Design**: Develop more quantum-native machine learning approaches
2. **Feature Engineering**: Create quantum feature maps better suited to specific data types
3. **Hybrid Methods**: Combine quantum and classical strengths effectively
4. **Benchmarking**: Establish fair comparison metrics for quantum vs classical methods

## 6. Technical Implementation Details

### 6.1 Quantum Circuit Design
- **Deutsch Algorithm**: Standard implementation with oracle variations
- **QSVM**: FidelityStatevectorKernel for noise-free simulation
- **Feature Maps**: ZZFeatureMap and PauliFeatureMap comparisons
- **State Analysis**: Comprehensive state vector evolution tracking

### 6.2 Performance Optimization
- **Kernel Precomputation**: Reduced QSVM computation time
- **Batch Processing**: Efficient sample handling
- **Memory Management**: Optimized for large kernel matrices
- **Visualization**: Real-time experiment monitoring

## 7. Future Work

### 7.1 Immediate Extensions
- Test additional quantum feature maps
- Explore different entanglement patterns
- Analyze noise effects systematically
- Compare with quantum kernel methods

### 7.2 Long-term Research
- Develop quantum-advantage-focused datasets
- Create new quantum ML algorithms
- Investigate quantum error correction impact
- Explore quantum federated learning

## 8. Acknowledgments

This work was conducted as part of a comprehensive quantum computing study, utilizing collaborative analysis with advanced AI systems to ensure thorough investigation and critical evaluation of results.

## 9. References

1. Deutsch, D. (1985). Quantum theory, the Church-Turing principle and the universal quantum computer. *Proceedings of the Royal Society of London A*, 400(1818), 97-117.

2. Havlíček, V., et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567(7747), 209-212.

3. Qiskit Documentation: https://qiskit.org/documentation/

4. Qiskit Machine Learning: https://qiskit-community.github.io/qiskit-machine-learning/

---

**Repository**: https://github.com/JinZhongcun/deutsch_algorithm_study

*This report serves as both an academic exercise and a practical guide for understanding the current state and future potential of quantum computing applications.*