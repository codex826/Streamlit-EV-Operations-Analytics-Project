# VoltRide AI Operations Dashboard ⚡🚗

An end-to-end Machine Learning and Data Analytics interactive dashboard built with Streamlit. This project was developed to solve the operational challenges presented in the **DecodeX 2026 Round 2: Business Case Study**. 

## 📖 Project Context
VoltRide, a mobility platform operating an exclusively electric vehicle (EV) fleet, faced a structural contradiction: despite steady growth in ride requests, customer satisfaction metrics were deteriorating due to sharp increases in ride cancellations and excessive waiting times.

Unlike conventional ride-hailing, operating an EV fleet introduces physical and temporal constraints, such as finite battery ranges and uneven charging infrastructure distribution. This application serves as a data-driven diagnostic tool to help management shift from reactive firefighting to proactive, analytics-led operations.

## ✨ Key Features
The application is divided into three core modules:

1. **Operational Dashboard (EDA)** 📊
   * Maps demand-supply stress across cities (Mumbai, Delhi, Bengaluru, Hyderabad) and zones.
   * Tracks conversion efficiency—the platform's ability to translate demand into completed rides under EV constraints.
   * Visualizes the overlap between charging congestion, battery levels, and ride outcomes.

2. **Cancellation Predictor (Machine Learning)** 🤖
   * Decomposes the drivers of ride cancellations using a **Random Forest Classifier**.
   * Analyzes multi-source triggers including system safeguards (low battery bands), customer impatience (wait times), and weather events (rainfall).
   * Provides a real-time prediction tool to assess the cancellation risk of a specific ride profile.

3. **Fleet Redeployment Engine** 🔄
   * Assesses fleet utilization efficiency across varying operational zones.
   * Visualizes an efficiency matrix (Wait Times vs. Cancellation Rates) to identify strong candidates for fleet redeployment without the need to increase total fleet size or incur heavy capital expenditure.

## 🛠️ Tech Stack
* **Frontend/Framework:** [Streamlit](https://streamlit.io)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (RandomForestClassifier)
* **Data Visualization:** Plotly Express

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
    link : (https://github.com/codex826/Streamlit-EV-Operations-Analytics-Project)

## Images: 

![image alt](https://github.com/codex826/Streamlit-EV-Operations-Analytics-Project/blob/main/Screenshot%202026-02-16%20122226.png?raw=true)
![image alt](https://github.com/codex826/Streamlit-EV-Operations-Analytics-Project/blob/main/Screenshot%202026-02-16%20114213.png?raw=true)
![image alt](https://github.com/codex826/Streamlit-EV-Operations-Analytics-Project/blob/main/Screenshot%202026-02-16%20114146.png?raw=true)
![image alt](https://github.com/codex826/Streamlit-EV-Operations-Analytics-Project/blob/main/Screenshot%202026-02-16%20114059.png?raw=true)
![image alt](https://github.com/codex826/Streamlit-EV-Operations-Analytics-Project/blob/main/Screenshot%202026-02-16%20113959.png?raw=true)



