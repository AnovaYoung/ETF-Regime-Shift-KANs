# Regime Detection in ETF Flows with Kolmogorov-Arnold Networks (KANs)

Welcome to the bleeding edge of financial machine learning research.

This project investigates the use of Kolmogorov-Arnold Networks (KANs) for detecting latent regime shifts in cross-asset ETF flows—an underexplored, high-potential application of a groundbreaking neural architecture.

---

## 🚀 Project Motivation

Traditional deep learning architectures (like MLPs and LSTMs) have made progress in modeling market regimes, but are often black-box and parameter-heavy.  
**Kolmogorov-Arnold Networks (KANs)**, inspired by the Kolmogorov-Arnold representation theorem, promise superior parameter efficiency and interpretability by employing learnable activation functions on edges.

The goal is to:
- **Test the limits** of KANs in financial, noisy, real-world settings.
- **Detect and anticipate** shifts between market states (e.g., "risk-on" and "risk-off") using cross-asset ETF flow data.
- **Advance the state of the art** and publish results that push both the ML and quant communities forward.

---

## 🧠 What are KANs?

KANs replace fixed neural activations with learnable, often spline-based, edge functions, allowing:
- More compact models with fewer parameters.
- Enhanced interpretability (key in finance).
- Flexibility to model complex, nonlinear relationships.

Read more in the [original KAN paper (arXiv)](https://arxiv.org/abs/2404.19756).

---

## 🔬 Tech Stack

- **Python 3.x**
- **PyKAN** (KAN implementation)  
- **Pandas / Numpy / Scipy** (data processing)
- **Matplotlib / Seaborn / Plotly** (visualization)
- **Jupyter Notebooks** (experiments and reporting)

---

## 📈 Data Sources

- **ETF Flow Data** (Bloomberg, Morningstar, or free sources if available)
- **Cross-asset prices** (Yahoo Finance, Quandl, or similar APIs)
- **Synthetic datasets** (for benchmarking and testing)

---

## 🗺️ Roadmap

### **1. Literature & Theory Review**
- [x] Survey all available literature and code on KANs
- [x] Review the state-of-the-art in financial regime detection
- [x] Document the project’s theoretical motivation

### **2. Data Acquisition & Preprocessing**
- [ ] Collect historical ETF flow and cross-asset price data
- [ ] Simulate synthetic datasets with known regime shifts
- [ ] Normalize, align, and clean time series data

### **3. KAN Prototyping & Benchmarking**
- [ ] Install and test PyKAN with toy datasets
- [ ] Build minimal regime-shift detection benchmark (e.g., logistic regression, LSTM)
- [ ] Develop KAN architecture for financial regime detection

### **4. Experimentation & Tuning**
- [ ] Tune KAN hyperparameters (grid size, width, depth, sparsification)
- [ ] Run experiments on real and synthetic data
- [ ] Compare KANs to standard models (MLP, LSTM, etc.)

### **5. Interpretability & Analysis**
- [ ] Prune and visualize learned functions for insight into regime detection
- [ ] Analyze when and why KANs outperform or fail
- [ ] Document findings on interpretability vs. accuracy tradeoff

### **6. Research Paper & Dissemination**
- [ ] Draft and submit a research paper (preprint or journal)
- [ ] Prepare an open-source code release and demo notebook
- [ ] Present results at meetups or academic venues

---

## 🤝 Contributing

PRs and issues are welcome! If you want to try KANs on new data or extend the benchmarking suite, open a discussion or pull request.

---

## 📚 References

- [Kolmogorov-Arnold Networks Paper (arXiv)](https://arxiv.org/abs/2404.19756)
- [PyKAN GitHub Repository](https://github.com/KindXiaoming/pykan)
- [Survey Paper on KANs](https://arxiv.org/abs/2411.06078)
