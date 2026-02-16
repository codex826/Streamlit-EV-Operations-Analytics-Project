# VoltRide AI Operations Dashboard ⚡🚗

An end-to-end Machine Learning and Data Analytics interactive dashboard built with Streamlit. [cite_start]This project was developed to solve the operational challenges presented in the **DecodeX 2026 Round 2: Business Case Study**[cite: 1, 2]. 

## 📖 Project Context
[cite_start]VoltRide, a mobility platform operating an exclusively electric vehicle (EV) fleet, faced a structural contradiction: despite steady growth in ride requests, customer satisfaction metrics were deteriorating due to sharp increases in ride cancellations and excessive waiting times[cite: 6, 8, 9].

[cite_start]Unlike conventional ride-hailing, operating an EV fleet introduces physical and temporal constraints, such as finite battery ranges and uneven charging infrastructure distribution[cite: 53, 54, 56]. [cite_start]This application serves as a data-driven diagnostic tool to help management shift from reactive firefighting to proactive, analytics-led operations[cite: 71].

## ✨ Key Features
The application is divided into three core modules:

1. **Operational Dashboard (EDA)** 📊
   * [cite_start]Maps demand-supply stress across cities (Mumbai, Delhi, Bengaluru, Hyderabad) and zones[cite: 31, 32, 126].
   * [cite_start]Tracks conversion efficiency—the platform's ability to translate demand into completed rides under EV constraints[cite: 70].
   * [cite_start]Visualizes the overlap between charging congestion, battery levels, and ride outcomes[cite: 89, 98].

2. **Cancellation Predictor (Machine Learning)** 🤖
   * [cite_start]Decomposes the drivers of ride cancellations using a **Random Forest Classifier**[cite: 131].
   * [cite_start]Analyzes multi-source triggers including system safeguards (low battery bands), customer impatience (wait times), and weather events (rainfall)[cite: 63, 78, 79].
   * Provides a real-time prediction tool to assess the cancellation risk of a specific ride profile.

3. **Fleet Redeployment Engine** 🔄
   * [cite_start]Assesses fleet utilization efficiency across varying operational zones[cite: 135].
   * [cite_start]Visualizes an efficiency matrix (Wait Times vs. Cancellation Rates) to identify strong candidates for fleet redeployment without the need to increase total fleet size or incur heavy capital expenditure[cite: 107, 136].

## 🛠️ Tech Stack
* **Frontend/Framework:** [Streamlit](https://streamlit.io)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (RandomForestClassifier)
* **Data Visualization:** Plotly Express

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
    link : (https://github.com/codex826/Streamlit-EV-Operations-Analytics-Project)

## Deployment and Link of the Project:
