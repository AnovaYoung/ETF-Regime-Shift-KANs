# Regime Detection in ETF Flows with Kolmogorov-Arnold Networks (KANs)

Welcome to the bleeding edge of financial machine learning research.

This project investigates the use of Kolmogorov-Arnold Networks (KANs) for detecting latent regime shifts in cross-asset ETF flows—an underexplored, high-potential application of a groundbreaking neural architecture.

---

## 🚀 Project Motivation

Traditional deep learning architectures (like MLPs and LSTMs) have made progress in modeling market regimes, but are often black-box and parameter-heavy.
**Kolmogorov-Arnold Networks (KANs)**, inspired by the Kolmogorov-Arnold representation theorem, promise superior parameter efficiency and interpretability by employing learnable activation functions on edges.

The goal is to:

* **Test the limits** of KANs in financial, noisy, real-world settings.
* **Detect and anticipate** shifts between market states (e.g., "risk-on" and "risk-off") using cross-asset ETF flow data, macro variables, and even market sentiment.
* **Advance the state of the art** and publish results that push both the ML and quant communities forward.

---

## 🧠 What are KANs?

KANs replace fixed neural activations with learnable, often spline-based, edge functions, allowing:

* More compact models with fewer parameters.
* Enhanced interpretability (key in finance).
* Flexibility to model complex, nonlinear relationships.

Read more in the [original KAN paper (arXiv)](https://arxiv.org/abs/2404.19756).

---

## 🔬 Tech Stack

* **Python 3.x**
* **PyKAN** (KAN implementation)
* **Dask** (for distributed data processing & big data merging)
* **Pandas / Numpy / Scipy** (data processing)
* **Pytrends** (Google Trends API for market sentiment)
* **Matplotlib / Seaborn / Plotly / Dash** (visualization, dashboarding)
* **Jupyter Notebooks & VS Code** (experiments and reporting)
* **(Planned)** Hugging Face Transformers, LLM APIs (for news/event features)

---

## 📈 Data Sources

* **ETF Flow Data** (Fund flow PDFs, converted to CSV; 40+ tickers)
* **ETF Daily Price Data** (Yahoo Finance, etc.)
* **Macro and Market Data** (VIX, Treasury rates, spreads, etc.)
* **Google Trends Data** (20+ market/regime keywords)
* **Synthetic datasets** (for benchmarking and testing)
* **(Planned)** Financial news/event data via APIs or LLMs

---

## 🗺️ Roadmap

### **1. Literature & Theory Review**

* [x] Survey all available literature and code on KANs
* [x] Review the state-of-the-art in financial regime detection
* [x] Document the project’s theoretical motivation

### **2. Data Acquisition & Preprocessing**

* [x] Collect historical ETF flow and cross-asset price data
* [x] Convert ETF flow PDFs to organized CSVs by ticker
* [x] Collect and save macro, treasury, and spread data
* [x] Collect and save Google Trends data for key market/regime keywords
* [ ] (Planned) Collect financial news and event features (LLM-based)
* [x] Organize all data in Google Drive, mirrored locally for VS Code
* [x] Inspect all files and date ranges for consistency
* [x] **Merge all data using Dask into a single, massive time-indexed DataFrame**
  **→ \[2025-07] Done! See “Current Status” below.**

### **3. KAN Prototyping & Benchmarking**

* [ ] Install and test PyKAN with toy datasets
* [ ] Build minimal regime-shift detection benchmark (e.g., logistic regression, LSTM)
* [ ] Develop KAN architecture for financial regime detection

### **4. Experimentation & Tuning**

* [ ] Tune KAN hyperparameters (grid size, width, depth, sparsification)
* [ ] Run experiments on real and synthetic data
* [ ] Compare KANs to standard models (MLP, LSTM, etc.)

### **5. Interpretability & Analysis**

* [ ] Prune and visualize learned functions for insight into regime detection
* [ ] Analyze when and why KANs outperform or fail
* [ ] Document findings on interpretability vs. accuracy tradeoff

### **6. Research Paper & Dissemination**

* [ ] Draft and submit a research paper (preprint or journal)
* [ ] Prepare an open-source code release and demo notebook
* [ ] Present results at meetups or academic venues

---

## ⚡️ Current Status (July 2025)

* **✅ Data acquisition, cleaning, and multi-source merging complete:**
  All ETF flows, price series, macro factors, and Google Trends data are merged into a single 5,000+ row x 284 column DataFrame in Parquet format.
* **✅ Ready for local development:**
  Dataset moved from GCP to a high-performance local machine (MSI 050US, RTX 5080, 64GB RAM) for rapid EDA, feature engineering, and model building.
* **⏳ KAN prototyping phase starting next:**
  PyKAN to be installed and first benchmark models to be developed.
* **⏳ (Planned):**
  Add LLM-powered news/event features, build live dashboards, and extend benchmarks.

---

## 🤝 Contributing

PRs and issues are welcome! If you want to try KANs on new data or extend the benchmarking suite, open a discussion or pull request.

---

## 📚 References

* [Kolmogorov-Arnold Networks Paper (arXiv)](https://arxiv.org/abs/2404.19756)
* [PyKAN GitHub Repository](https://github.com/KindXiaoming/pykan)
* [Survey Paper on KANs](https://arxiv.org/abs/2411.06078)

---
