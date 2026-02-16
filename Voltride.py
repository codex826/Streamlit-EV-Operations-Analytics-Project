import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. SETUP & SYNTHETIC DATA GENERATION ---
st.set_page_config(page_title="VoltRide AI Operations", layout="wide")

@st.cache_data
def generate_voltride_data(n_samples=5000):
    """Generates synthetic data mirroring the VoltRide Case Study."""
    np.random.seed(42)
    
    cities = ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad']
    zones = ['Zone A (Commercial)', 'Zone B (Residential)', 'Zone C (Transit Hub)']
    weather_conditions = ['Clear', 'Rain']
    
    data = {
        'City': np.random.choice(cities, n_samples),
        'Zone': np.random.choice(zones, n_samples),
        'Hour_of_Day': np.random.randint(0, 24, n_samples),
        'Weather': np.random.choice(weather_conditions, n_samples, p=[0.8, 0.2]),
        'Battery_Level_At_Pickup': np.random.uniform(10, 100, n_samples),
        'Surge_Multiplier': np.random.uniform(1.0, 3.0, n_samples),
        'Charging_Wait_Time_Mins': np.random.uniform(0, 60, n_samples),
        'Base_Fare_INR': np.random.uniform(150, 600, n_samples) # Added base fare in INR
    }
    
    df = pd.DataFrame(data)
    
    # Financial Calculation: Estimated Fare = Base Fare * Surge Multiplier
    df['Estimated_Fare_INR'] = df['Base_Fare_INR'] * df['Surge_Multiplier']
    
    # Logic to simulate cancellations
    def determine_outcome(row):
        cancel_prob = 0.1 
        
        if row['Hour_of_Day'] in [8, 9, 10, 18, 19, 20]:
            cancel_prob += 0.2
            
        if row['Battery_Level_At_Pickup'] < 25:
            cancel_prob += 0.3
            
        if row['Weather'] == 'Rain':
            cancel_prob += 0.15
            
        if row['Charging_Wait_Time_Mins'] > 30:
            cancel_prob += 0.15
            
        return 'Cancelled' if np.random.rand() < cancel_prob else 'Completed'

    df['Ride_Outcome'] = df.apply(determine_outcome, axis=1)
    df['Is_Cancelled'] = df['Ride_Outcome'].apply(lambda x: 1 if x == 'Cancelled' else 0)
    
    # Financial Calculations: Realized vs. Lost Revenue
    df['Revenue_Realized_INR'] = df.apply(lambda row: row['Estimated_Fare_INR'] if row['Ride_Outcome'] == 'Completed' else 0, axis=1)
    df['Revenue_Lost_INR'] = df.apply(lambda row: row['Estimated_Fare_INR'] if row['Ride_Outcome'] == 'Cancelled' else 0, axis=1)
    
    return df

df = generate_voltride_data()

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("VoltRide OS")
st.sidebar.markdown("Electric Mobility Analytics")
page = st.sidebar.radio("Navigation", [
    "1. Operational Dashboard", 
    "2. Cancellation Predictor (ML)", 
    "3. Fleet Redeployment",
    "4. Financial Impact Analysis"
])

# --- 3. PAGE 1: OPERATIONAL DASHBOARD ---
if page == "1. Operational Dashboard":
    st.title("Demand-Supply Stress Mapping")
    st.write("Analyze structural inefficiencies across cities and time windows.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Ride Requests", len(df))
    col2.metric("Overall Conversion Efficiency", f"{(len(df[df['Ride_Outcome'] == 'Completed']) / len(df)) * 100:.1f}%")
    col3.metric("Avg Charging Wait Time", f"{df['Charging_Wait_Time_Mins'].mean():.1f} mins")

    st.subheader("Cancellation Rates by Zone and Hour")
    stress_map = df.groupby(['Zone', 'Hour_of_Day'])['Is_Cancelled'].mean().reset_index()
    fig1 = px.density_heatmap(stress_map, x='Hour_of_Day', y='Zone', z='Is_Cancelled', 
                              color_continuous_scale='Reds',
                              title="Heatmap of Ride Cancellations (Higher is Worse)")
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Impact of Battery Constraints vs Wait Times")
    fig2 = px.scatter(df, x='Battery_Level_At_Pickup', y='Charging_Wait_Time_Mins', color='Ride_Outcome',
                      opacity=0.6, title="Ride Outcomes based on Battery and Charging Queues")
    st.plotly_chart(fig2, use_container_width=True)

# --- 4. PAGE 2: CANCELLATION PREDICTOR (ML) ---
elif page == "2. Cancellation Predictor (ML)":
    st.title("Cancellation Driver Decomposition")
    st.write("Diagnose the root causes of cancellations using a Random Forest Classifier.")
    
    features = ['Hour_of_Day', 'Battery_Level_At_Pickup', 'Surge_Multiplier', 'Charging_Wait_Time_Mins']
    X = df[features].copy()
    X['Is_Rain'] = df['Weather'].apply(lambda x: 1 if x == 'Rain' else 0)
    y = df['Is_Cancelled']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.success(f"Model trained with an accuracy of {accuracy*100:.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', title="Drivers of Cancellation")
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col2:
        st.subheader("Predict a Ride Outcome")
        with st.form("predict_form"):
            input_hour = st.slider("Hour of Day", 0, 23, 8)
            input_battery = st.slider("Battery Level (%)", 0, 100, 20)
            input_surge = st.slider("Surge Multiplier", 1.0, 3.0, 1.5)
            input_wait = st.slider("Charging Wait Time (Mins)", 0, 60, 45)
            input_weather = st.selectbox("Weather", ["Clear", "Rain"])
            
            submit = st.form_submit_button("Predict Outcome")
            
            if submit:
                is_rain = 1 if input_weather == "Rain" else 0
                input_data = np.array([[input_hour, input_battery, input_surge, input_wait, is_rain]])
                prob = model.predict_proba(input_data)[0][1]
                
                if prob > 0.5:
                    st.error(f"High Risk of Cancellation! Probability: {prob*100:.1f}%")
                else:
                    st.success(f"Ride likely to complete. Cancellation Risk: {prob*100:.1f}%")

# --- 5. PAGE 3: FLEET REDEPLOYMENT ---
elif page == "3. Fleet Redeployment":
    st.title("Fleet Utilization Efficiency")
    st.write("Identify operational imbalances to shift fleet capacity without increasing total fleet size.")
    
    st.markdown("""
    **Analytical Logic**: 
    If a zone has a low cancellation rate but high charging wait times, it is over-supplied and congested. 
    If a zone has a high cancellation rate but low charging wait times, it is under-supplied.
    """)
    
    zone_stats = df.groupby('Zone').agg(
        Cancel_Rate=('Is_Cancelled', 'mean'),
        Avg_Wait=('Charging_Wait_Time_Mins', 'mean'),
        Total_Requests=('Ride_Outcome', 'count')
    ).reset_index()
    
    fig3 = px.scatter(zone_stats, x='Avg_Wait', y='Cancel_Rate', size='Total_Requests', color='Zone',
                      title="Zone Efficiency Matrix (Wait Times vs Cancellations)")
    
    fig3.add_hline(y=zone_stats['Cancel_Rate'].mean(), line_dash="dash", line_color="white")
    fig3.add_vline(x=zone_stats['Avg_Wait'].mean(), line_dash="dash", line_color="white")
    
    st.plotly_chart(fig3, use_container_width=True)
    
    st.info("💡 **Recommendation**: Redeploy vehicles from zones in the bottom-right quadrant (High Wait, Low Cancellations) to the top-left quadrant (Low Wait, High Cancellations).")

# --- 6. PAGE 4: FINANCIAL IMPACT ANALYSIS ---
elif page == "4. Financial Impact Analysis":
    st.title("Financial Impact & Revenue Leakage")
    st.write("Quantifying the business cost of operational inefficiencies and ride cancellations.")

    # High-level Financial KPIs
    total_potential = df['Estimated_Fare_INR'].sum()
    total_realized = df['Revenue_Realized_INR'].sum()
    total_lost = df['Revenue_Lost_INR'].sum()
    loss_percentage = (total_lost / total_potential) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Potential Demand (INR)", f"₹{total_potential:,.0f}")
    col2.metric("Realized Revenue (INR)", f"₹{total_realized:,.0f}")
    col3.metric("Revenue Leakage (INR)", f"₹{total_lost:,.0f}")
    col4.metric("Leakage %", f"{loss_percentage:.1f}%")

    # Visualizing the Financial Impact
    st.subheader("Revenue Leakage by Zone")
    revenue_zone = df.groupby('Zone').agg(
        Total_Potential=('Estimated_Fare_INR', 'sum'),
        Revenue_Lost=('Revenue_Lost_INR', 'sum')
    ).reset_index()
    
    # Calculate % loss per zone for better context
    revenue_zone['Loss_Percentage'] = (revenue_zone['Revenue_Lost'] / revenue_zone['Total_Potential']) * 100
    
    fig4 = px.bar(revenue_zone, x='Zone', y='Revenue_Lost', 
                  title="Lost Revenue Due to Cancellations by Operational Zone (INR)",
                  text=revenue_zone['Revenue_Lost'].apply(lambda x: f"₹{x:,.0f}"),
                  color='Loss_Percentage', color_continuous_scale='Reds')
    st.plotly_chart(fig4, use_container_width=True)

    # Time of day financial analysis
    st.subheader("Hourly Financial Stress Mapping")
    hourly_finance = df.groupby('Hour_of_Day')['Revenue_Lost_INR'].sum().reset_index()
    fig5 = px.line(hourly_finance, x='Hour_of_Day', y='Revenue_Lost_INR', 
                   title="Revenue Lost by Hour of Day (Peak vs Off-Peak)",
                   markers=True)
    st.plotly_chart(fig5, use_container_width=True)
    
    st.warning("⚠️ **Managerial Insight**: The steepest revenue drops occur during periods of high surge pricing combined with high charging wait times. Optimizing charging schedules prior to these peak hours directly impacts the bottom line.")