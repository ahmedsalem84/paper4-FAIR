# TrustFair-FL: Reconciling Byzantine Fault Tolerance and Fairness in Personalized Federated IoT Intrusion Detection via Semantic Gradient Auditing

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/🤗%20Transformers-DistilBERT%20%7C%20MiniLM-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Submission Ready](https://img.shields.io/badge/MDPI%20MAKE-Peer%20Review-brightgreen.svg)]()

Official implementation and experimental artifact repository for the research paper:  
**"TrustFair-FL: Reconciling Byzantine Fault Tolerance and Fairness in Personalized Federated IoT Intrusion Detection via Semantic Gradient Auditing"**  
*Abdullah AlHayan and Jalal Al-Muhtadi (King Saud University)*

---

## 📖 Executive Summary

Federated Learning (FL) provides a decentralized, privacy-preserving paradigm for Collaborative Intrusion Detection Systems (IDSs) in the Internet of Things (IoT). However, deploying FL in real-world heterogeneous environments exposes **The Byzantine-Fairness Dilemma**:
1. **Robustness Destroys Fairness:** Classical Byzantine-resilient aggregators (e.g., Krum, Multi-Krum, Coordinate-wise Median) rely on geometric outlier filtering. Under extreme non-IID data skew, legitimate minority clusters (e.g., low-volume specialized edge sensors) naturally diverge from the dominant centroid and are discarded as anomalies, exacerbating critical security blind spots.
2. **Fairness Invites Model Hijacking:** Unconstrained fairness-aware aggregators (e.g., q-FFL, FairPFL) dynamically amplify high-loss clients. Byzantine adversaries exploit this mechanism by submitting degraded, poisoned updates, capturing up to **79.00%** of the global aggregation weight.

**TrustFair-FL** resolves this conflict by decoupling directional/semantic verification from fairness reweighting through a **Dual-Gated Aggregation Architecture**:
* **Contextual Semantic Gradient Auditing:** Serializes categorical flow parameters (`protocol`, `service`, `flags`, `state`) into linguistic sequences via compact language models (`DistilBERT` / `MiniLM-L6`) to audit updates on unlabelled server reference probes.
* **Three-Stage Gating Pipeline:**
  * **Gate 1 (Semantic Verification):** Directional cosine alignment ($S_C \ge 0.30$), norm ceiling ($\nu_k \le 3.0$), and functional signature tracking ($S_{\text{sem}} \ge 0.30$).
  * **Gate 2 (Temporal Reputation):** Exponential Moving Average (EMA) scoring ($R_k^{(t)} \ge 0.50$, $\lambda = 0.85$) with adaptive grace recovery against intermittent/sleeper adversaries.
  * **Gate 3 (Trust-Conditioned Dead-Zone Fairness):** Restricts fairness amplification ($\Phi_k^{(t)}$) strictly to verified honest clusters while truncating adversary multipliers to zero.
* **Split Personalization:** Trains a shared temporal backbone (BiLSTM + Multi-Head Attention) while retaining private classification heads locally to guarantee on-device edge inference autonomy even if isolated by the server.

---

## 🏛️ System Architecture
[ Distributed IoT Network: Heterogeneous Clusters ]
         ┌───────────────────────┬───────────────────────┐
         ▼                       ▼                       ▼
[Cluster 1: Head Gateway] [Cluster 6: Honest Minor.] [Cluster 9: Byzantine]
(High-Volume Traffic)     (Low-Volume Sensor Pod)   (Label/Sign Poisoning)
         │                       │                       │
         ▼                       ▼                       ▼
 Local Feature Space     Local Feature Space     Local Feature Space
         │                       │                       │
         └───────────────┬───────┴───────────────────────┘
                         ▼
          [Contextual Transformer Tokenizer]
          (protocol: tcp, service: http, flag: SF, state: ESTABLISHED)
                         │
                         ▼
          [Split-Personalized Local Training]
          • Shared Backbone : w_k (BiLSTM + Temporal Attention)
          • Personal Head   : θ_k (Locally Retained MLP)
                         │
                         ▼ (Candidate Uplink Updates: Δw_k)

                         ---

## 📊 Key Experimental Findings

All experiments evaluate **10 heterogeneous device clusters** over **35 communication rounds** across **3 random seeds** ($\mathcal{S}=\{11, 22, 33\}$) on two major benchmarks:

### 1. Primary Benchmark: NSL-KDD Multi-Ratio Sweep ($0\%$ to $40\%$ Poisoning)
* **Resilience Under Extreme Attack ($\rho = 40\%$):** Standard `FedAvg` degrades to **$26.22\%$** Macro-F1; unconstrained `FairPFL` and `q-FFL` collapse to **$25.47\%$** and **$24.74\%$** (ceding up to $79.00\%$ weight to adversaries). **TrustFair-FL maintains a stable $61.41\%$ Macro-F1**, outperforming `FedAvg` by $+35.19$ percentage points ($p = 0.1106$, Cohen's $d_z = 1.59$).
* **Minority Protection:** `TrustFair-FL` sustains a worst-client lower-tail floor of **$36.37\%$** and Bottom-20% floor of **$40.84\%$**, whereas `FedAvg` and `FairPFL` degrade below $17\%$.
* **Minority Amplification:** Delivers **$2.36\times$** amplification above natural sample share for Client $C_6$ in clean operating environments.
* **Byzantine Suppression:** Caps malicious weight integration to **$\le 15.48\%$**, completely mitigating the model hijacking observed in conventional fairness frameworks.

### 2. Cross-Dataset Generalization: Edge-IIoTset (15 Classes, Industrial IoT)
* **Complete Adversarial Isolation:** `TrustFair-FL` achieves **$0.00\%$ Byzantine influence across all threat ratios ($0\%$ to $40\%$)**, completely blocking poisoned gradients while `FairPFL` absorbs up to $18.71\%$ malicious weight and coordinate-wise baselines (`Trimmed-Mean`, `Median`) suffer up to **$40.00\%$ coordinate retention**.
* **Tightest Fairness Gap:** Achieves the lowest inter-cluster fairness gap among all methods ($\Delta F1 = 0.1428$, Gini = $0.1477$).

### 3. Omniscient Adaptive Attacks (ALIE, Min-Max, IPM, Norm-Mimic, F1-Liar)
* **Zero Vulnerability to IPM:** Under Inner Product Manipulation, `TrustFair-FL` enforces **$0.00\%$ Byzantine influence**.
* **Telemetry Lying Immunity:** Under client-reported F1 manipulation (adversaries reporting false $F1=0.05$), `FairPFL` falls to $28.40\%$, while `TrustFair-FL` maintains **$60.18\%$ Macro-F1** by directly auditing gradient trajectories.

---

## 📁 Repository Structure

```text
TrustFair-FL/
├── tables/                                     # Raw empirical CSV tables (Tables 1 - 10)
│   ├── table1_byzantine_ratio_sweep.csv        # Primary NSL-KDD ratio sweep (0% to 40%)
│   ├── table1b_edge_iiotset_ratio_sweep.csv    # Cross-dataset Edge-IIoTset benchmark
│   ├── table2_clean_vs_attack_degradation.csv  # Drop from clean baseline across all metrics
│   ├── table3_byzantine_detector_metrics.csv   # Dual-gate TPR, FPR, FNR, confusion counts
│   ├── table4_minority_vs_malicious_gate_decisions.csv # Controlled Client 6 vs. 9 audit
│   ├── table5_representation_ablation.csv      # DistilBERT vs. MiniLM vs. One-Hot
│   ├── table6_local_isolation_autonomy.csv     # On-device inference under server isolation
│   ├── table7_tost_clean_equivalence.csv       # Two One-Sided Tests (TOST) analysis
│   ├── table8_component_ablations.csv          # Ablations (noG1, noRep, trustOnly, etc.)
│   ├── table9_minority_weight.csv              # Client 6 aggregation weight vs. sample share
│   └── table10_adaptive_attacks.csv            # ALIE, Min-Max, IPM, Norm-Mimic, F1-Liar
├── plots/                                      # Publication-ready figures (300 DPI)
│   ├── fig1_cross_dataset_macro_f1.png         # Dual-panel cross-dataset convergence
│   ├── fig2_worst_client_f1_vs_ratio.png       # Lower-tail robustness floor
│   ├── fig3_fairness_gap_vs_ratio.png          # Inter-cluster fairness gap dynamics
│   ├── fig4_byzantine_influence_vs_ratio.png   # Adversarial parameter influence (with >=50% zone)
│   ├── fig5_targeted_label_flip_asr.png        # Backdoor Attack Success Rate (ASR)
│   ├── fig6_detection_tpr_fpr_fnr.png          # Dual-gate screening error curves
│   ├── fig7_controlled_c6_vs_c9_trajectory.png # Controlled 35-round tracking streams
│   └── fig8_minority_weight_vs_ratio.png       # Minority amplification vs. suppression zones
├── main_benchmark_runner.py                    # Master execution script (NSL-KDD deep dive)
├── edge_iiotset_runner.py                      # Standalone GPU runner for Edge-IIoTset
├── generate_publication_plots.py               # Standalone 300-DPI figure generator
├── requirements.txt                            # Pinned Python package dependencies
├── reproducibility_manifest.json               # Hardware, seeds, and execution checksums
└── README.md                                   # This repository documentation

Cryptographic Dataset Provenance

To guarantee data integrity and eliminate data snooping, datasets are audited before partitioning. Preprocessing scalers are fitted strictly on client training splits (verified by an 8-point mathematical isolation audit):
Benchmark Dataset	Raw Rows	Deduplicated Usable Rows	Retention	Cryptographic SHA-256 Digest
NSL-KDD (Combined Train+Test)	

        
148,517
148,517

      

	

        
138,394
138,394

      

	

        
93.18%
93.18%

      

	Train: e8a3b5c1... | Test: 9f201d4a...
Edge-IIoTset (DNN-Ready)	

        
221,920
221,920

      

	

        
215,610
215,610

      

	

        
97.16%
97.16%

      

	f4b829cc0e1189db878340d...
🚀 Quick Start & Reproduction Guide
1. Environment Installation

Clone the repository and set up a clean Python 3.10+ virtual environment:
code Bash

git clone https://github.com/445107907/TrustFair-FL.git
cd TrustFair-FL

conda create -n trustfair python=3.10 -y
conda activate trustfair
pip install -r requirements.txt

2. Dataset Preparation

    NSL-KDD: Automatically downloaded, verified, and cached on first run.

    Edge-IIoTset: Download the official DNN-EdgeIIoT-dataset.csv from Western-OC2 Lab and place it in the root directory:
    code Bash

    # Ensure the real CSV (~1.16 GB) is in the root directory:
    ls -lh DNN-EdgeIIoT-dataset.csv

3. Running the Primary Benchmark Suite (NSL-KDD)

Executes the multi-ratio sweep, adaptive attacks, ablations, and TOST analysis across seeds [11, 22, 33]:
code Bash

python main_benchmark_runner.py

4. Running the Standalone Edge-IIoTset Benchmark

Executes the 8 protocols across all 5 Byzantine ratios on industrial IoT traffic (runs in

        
≈10
≈10

      

–

        
15
15

      

minutes on a modern GPU):
code Bash

python edge_iiotset_runner.py

5. Regenerating Publication Figures

Reads directly from the generated CSV tables and outputs high-resolution 300-DPI plots into ./plots/ with zero legend collisions:
code Bash

python generate_publication_plots.py

🔬 Evaluated Defense Baselines
Protocol	Strategy	Outlier Filtering	Fairness Reweighting	Split Personalization
FedAvg	Data-proportional averaging	❌	❌	❌
Krum	Geometric distance minimization	✅	❌	❌
Multi-Krum	Top-

        
m
m

      

candidate vector averaging	✅	❌	❌
Trimmed-Mean	Coordinate-wise scalar trimming	✅	❌	❌
Median	Coordinate-wise scalar median	✅	❌	❌
q-FFL	Static loss-exponent reweighting	❌	✅	❌
FairPFL	Adaptive dead-zone fairness	❌	✅	✅
TrustFair-FL (Ours)	Dual-Gated Trust + Dead-Zone Fairness	✅	✅	✅
📑 Citation & Academic Attribution

If you use this codebase, empirical tables, or the TrustFair-FL framework in your research, please cite our publications:
code Bibtex

@article{alhayan2026trustfair,
  author    = {Abdullah AlHayan and Jalal Al-Muhtadi},
  title     = {TrustFair-FL: Reconciling Byzantine Fault Tolerance and Fairness in Personalized Federated IoT Intrusion Detection via Semantic Gradient Auditing},
  journal   = {Machine Learning and Knowledge Extraction},
  year      = {2026}
}

@article{alhayan2026fidmf,
  author    = {Abdullah AlHayan and Jalal Al-Muhtadi},
  title     = {Federated learning-powered real-time behavioral intrusion detection leveraging LSTM, attention, GANs, and large language models},
  journal   = {Scientific Reports},
  volume    = {16},
  number    = {1},
  pages     = {10172},
  year      = {2026},
  doi       = {10.1038/s41598-026-40763-5}
}

@article{alhayan2025stl,
  author    = {Abdullah AlHayan and Jalal Al-Muhtadi},
  title     = {A Hybrid STL-Deep Learning Framework for Behavioral-Based Intrusion Detection in IoT Environments},
  journal   = {Applied Sciences},
  volume    = {15},
  number    = {12},
  pages     = {6421},
  year      = {2025},
  doi       = {10.3390/app15126421}
}

⚖️ License

This project is licensed under the MIT License — see the LICENSE file for details.
🤝 Acknowledgments

This research was supported by the College of Computer and Information Sciences (CCIS) and the Center of Excellence in Information Assurance (CoEIA) at King Saud University, Riyadh, Saudi Arabia.
