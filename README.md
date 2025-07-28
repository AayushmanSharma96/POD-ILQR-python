# POD‑iLQR‑python 📈⚙️  
*Information‑state model‑based RL for nonlinear PDE control*

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)  

This repo contains a **Python implementation of POD‑iLQR (POD²C)**—an information‑state extension of model‑based reinforcement learning extended to high‑dimensional PDE systems.

> **Reference paper:**  
> *“An Information‑State Based Approach to Linear Time‑Varying System Identification and Control”*  
> [[arXiv 2211.10583](https://arxiv.org/pdf/2211.10583)]

---

## 🚀 Key ideas
* **Information‑state dynamics**: latent dynamics updated online using measurement residuals.
* **ARMA-LTV system identification**: The original state-space is recovered by stacking past information state and control inputs to fit a linear, time-varying ARMA model.  
* **iLQR in the loop**: trajectory optimization in the reduced space drives real‑time control.  

---

## 🏗️ Applications included

| PDE / System | State Dim. | Observations | Control Inputs |
|--------------|-----------:|-------------:|---------------:|
| **1‑D viscous Burgers** | 100 | 3 probes | 2 (end‑wall suction / blowing) |
| **Allen‑Cahn** (micro‑structure phase separation) | 2 500 | 5 probes | 4 control inputs |
| **Cahn‑Hilliard** (micro‑structure phase separation) | 400 | 5 probes | 4 control inputs |

> For scripts, see `burgers_model_free_ddp.py`, `material_model_free_ddp.py`.

---

## 📦 Installation

```bash
git clone https://github.com/AayushmanSharma96/POD-ILQR-python.git
cd POD-ILQR-python


