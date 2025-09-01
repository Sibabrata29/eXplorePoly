# PolyeXplore : Polymer Exploration Model

An interactive **Streamlit web app** that visualizes how **polymer structural features** influence their material properties.  
Developed as an educative and exploratory tool for structure–property relationships in polymers.

🔗 **Live App:** [https://eXplorePoly.onrender.com](https://eXplorePoly.onrender.com)

---

## 🌟 Overview
PolyeXplore connects **repeat unit features** of polymers to **key material properties** through interactive visualizations.

It complements fundamental *Group Contribution* approaches (Mark, van Krevelen, Bicerano) by providing a **macro-level influence map** that highlights:
- Which structural features (flexibility, aromaticity, H-bonding, etc.) drive material performance.
- How polymer categories (commodity, engineering, high-performance) compare.
- Visual exploration of feature–property correlations.

---

## 📊 Key Features

- **Structural Feature Input**  
  Quantify repeat unit features and compare against dataset values.

- **3D Molecular Visualization**  
  View polymer repeat units interactively with **RDKit + py3Dmol**.

- **Molecular Descriptor Analysis**  
  Descriptor comparison with a polyethylene reference.

- **Nearest Equivalent Polymers**  
  Find polymers closest in property index space.

- **Property Hierarchy Heatmaps**  
  Explore properties across commodity, engineering, and high-performance categories.

- **Randomized Insights**  
  On each refresh, view different subsets of feature–property maps.

- **Correlation & Influence Maps**  
  - Random subset sign maps (positive/negative/neutral correlations).  
  - Top 3 features influencing selected properties.  
  - Positive vs Negative contribution breakdown.

---

## 🛠️ Installation (Local)

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

Run the app:

```bash
streamlit run neweXplore.py


---


