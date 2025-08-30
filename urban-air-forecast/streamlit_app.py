import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Urban Air Quality Policy Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .aqi-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .policy-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .urgency-critical { background-color: #feb2b2; color: #742a2a; padding: 0.25rem 0.5rem; border-radius: 12px; }
    .urgency-high { background-color: #fbb6ce; color: #702459; padding: 0.25rem 0.5rem; border-radius: 12px; }
    .urgency-medium { background-color: #fed7d7; color: #742a2a; padding: 0.25rem 0.5rem; border-radius: 12px; }
    .urgency-low { background-color: #c6f6d5; color: #22543d; padding: 0.25rem 0.5rem; border-radius: 12px; }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

class PolicyAssistant:
    def __init__(self):
        self.aqi_categories = [
            {"category": "Good", "range": "0-50", "color": "green", "urgency": "low", "descriptor": "Air quality is satisfactory; no interventions needed."},
            {"category": "Moderate", "range": "51-100", "color": "yellow", "urgency": "low", "descriptor": "Acceptable; some pollutants may be a concern for sensitive groups."},
            {"category": "Unhealthy_for_Sensitive_Groups", "range": "101-150", "color": "orange", "urgency": "medium", "descriptor": "Sensitive individuals at risk."},
            {"category": "Unhealthy", "range": "151-200", "color": "red", "urgency": "high", "descriptor": "Everyone may begin to experience health effects."},
            {"category": "Very_Unhealthy", "range": "201-300", "color": "purple", "urgency": "critical", "descriptor": "Health alert: emergency conditions."},
            {"category": "Hazardous", "range": "301-500", "color": "maroon", "urgency": "critical", "descriptor": "Serious health effects for entire population."}
        ]
        
        self.policy_templates = {
            "traffic": [
                {"intervention": "Restrict heavy vehicles during peak hours.", "expected_impact": "Reduce PM2.5 and NO2"},
                {"intervention": "Encourage carpooling and public transport.", "expected_impact": "Minor local AQI improvement"}
            ],
            "industry": [
                {"intervention": "Mandate temporary emission caps.", "expected_impact": "Significant SO2/NO2 reduction"},
                {"intervention": "Schedule surprise inspections within 48 hours.", "expected_impact": "Minor improvement, increases compliance"}
            ],
            "dust": [
                {"intervention": "Street watering and dust nets on construction sites.", "expected_impact": "Reduce PM10"}
            ],
            "public_health": [
                {"intervention": "Issue health advisory for vulnerable groups.", "expected_impact": "Protects sensitive groups"},
                {"intervention": "Distribute masks in schools and hospitals.", "expected_impact": "Protective impact on exposure, not AQI"}
            ],
            "general": [
                {"intervention": "Coordinate city emergency response.", "expected_impact": "Ensures rapid enforcement"},
                {"intervention": "Public communication via radio/SMS/TV.", "expected_impact": "Increases compliance"}
            ]
        }
        
        self.expected_impact_guidance = {
            "qualitative": ["Minor improvement in local AQI", "Noticeable dust reduction", "Significant improvement if enforced strictly"],
            "quantitative": [
                {"action": "traffic_restriction", "value": 15, "unit": "% PM2.5 reduction"},
                {"action": "dust_control", "value": 10, "unit": "% PM10 reduction"},
                {"action": "industrial_emission_cap", "value": 20, "unit": "% SO2/NO2 reduction"},
                {"action": "public_health_advisory", "value": 0, "unit": "Direct AQI impact"}
            ]
        }

    def get_aqi_category(self, aqi_value):
        for category in self.aqi_categories:
            min_val, max_val = map(int, category["range"].split('-'))
            if min_val <= aqi_value <= max_val:
                return category
        return self.aqi_categories[-1]  # Default to Hazardous if out of range

    def generate_policies(self, input_data):
        aqi_value = input_data["forecast_summary"]["predicted_aqi_value"]
        aqi_category = self.get_aqi_category(aqi_value)
        
        # Normalize drivers
        drivers = {
            "traffic": "high" if input_data["drivers_summary"]["traffic"] == "heavy" else input_data["drivers_summary"]["traffic"],
            "dust": input_data["drivers_summary"]["dust"],
            "industry": input_data["drivers_summary"]["industry"],
            "weather_condition": input_data["drivers_summary"]["weather_condition"],
            "seasonal_factor": input_data["drivers_summary"]["seasonal_factor"]
        }
        
        policies = []
        
        # Generate policies based on drivers
        for category, templates in self.policy_templates.items():
            if category == "traffic" and drivers["traffic"] in ["high", "medium"]:
                for template in templates:
                    policies.append(self._create_policy(category, template, aqi_category, drivers, aqi_value))
            elif category == "dust" and drivers["dust"] in ["high", "medium"]:
                for template in templates:
                    policies.append(self._create_policy(category, template, aqi_category, drivers, aqi_value))
            elif category == "industry" and drivers["industry"] in ["high", "medium"]:
                for template in templates:
                    policies.append(self._create_policy(category, template, aqi_category, drivers, aqi_value))
            elif category == "public_health" and aqi_value > 100:
                for template in templates:
                    policies.append(self._create_policy(category, template, aqi_category, drivers, aqi_value))
            elif category == "general" and aqi_category["urgency"] == "critical":
                for template in templates:
                    policies.append(self._create_policy(category, template, aqi_category, drivers, aqi_value))
        
        return {
            "station_id": input_data["station_id"],
            "timestamp": input_data["timestamp"],
            "aqi": {
                "value": aqi_value,
                "category": aqi_category["category"],
                "color": aqi_category["color"],
                "urgency": aqi_category["urgency"],
                "descriptor": aqi_category["descriptor"]
            },
            "drivers": drivers,
            "policies": policies
        }

    def _create_policy(self, category, template, aqi_category, drivers, aqi_value):
        # Get quantitative impact
        action_map = {
            "traffic": "traffic_restriction",
            "dust": "dust_control", 
            "industry": "industrial_emission_cap",
            "public_health": "public_health_advisory",
            "general": "public_health_advisory"
        }
        
        quant_impact = next((q for q in self.expected_impact_guidance["quantitative"] 
                           if q["action"] == action_map.get(category, "public_health_advisory")), 
                          {"value": 0, "unit": "Direct AQI impact"})
        
        return {
            "category": category,
            "intervention": template["intervention"],
            "expected_impact": {
                "qualitative": "Significant improvement if enforced strictly" if "Restrict" in template["intervention"] or "Mandate" in template["intervention"] else "Minor improvement in local AQI",
                "quantitative": quant_impact
            },
            "urgency": aqi_category["urgency"],
            "rationale": f"{category.title()} intervention needed due to {aqi_category['category']} AQI ({aqi_value}) with {drivers['weather_condition']} weather conditions."
        }

def main():
    # Initialize session state
    if 'policy_assistant' not in st.session_state:
        st.session_state.policy_assistant = PolicyAssistant()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Urban Air Quality Policy Assistant</h1>
        <p>Generate actionable policy recommendations based on air quality forecasts and pollution drivers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📊 Input Parameters")
        
        # Station Information
        st.subheader("🏢 Station Information")
        station_id = st.text_input("Station ID", value="STN_DELHI_001")
        
        col1, col2 = st.columns(2)
        with col1:
            city = st.text_input("City", value="Delhi")
            latitude = st.number_input("Latitude", value=28.6139, min_value=-90.0, max_value=90.0, step=0.0001)
        with col2:
            state = st.text_input("State", value="Delhi")
            longitude = st.number_input("Longitude", value=77.2090, min_value=-180.0, max_value=180.0, step=0.0001)
        
        country = st.text_input("Country", value="India")
        
        # AQI Forecast
        st.subheader("🌡️ AQI Forecast")
        aqi_value = st.slider("Predicted AQI Value", min_value=0, max_value=500, value=275, step=1)
        
        # Real-time AQI category display
        aqi_category = st.session_state.policy_assistant.get_aqi_category(aqi_value)
        st.markdown(f"""
        <div style="background-color: {aqi_category['color']}; color: white; padding: 0.5rem; border-radius: 5px; text-align: center; margin: 0.5rem 0;">
            <strong>{aqi_category['category'].replace('_', ' ')}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        confidence = st.slider("Confidence", min_value=0.0, max_value=1.0, value=0.87, step=0.01)
        time_horizon = st.number_input("Time Horizon (hours)", min_value=1, max_value=168, value=48)
        
        # Pollution Drivers
        st.subheader("🚗 Pollution Drivers")
        traffic = st.selectbox("Traffic Level", ["low", "medium", "high"], index=2)
        dust = st.selectbox("Dust Level", ["low", "medium", "high"], index=2)
        industry = st.selectbox("Industrial Activity", ["low", "medium", "high"], index=1)
        weather = st.selectbox("Weather Condition", ["clear", "windy", "rainy", "foggy", "stagnant"], index=4)
        seasonal = st.selectbox("Seasonal Factor", ["none", "harvest_burning", "festival_fireworks", "winter_inversion"], index=3)
        
        # Metadata
        st.subheader("📋 Metadata")
        data_source = st.text_input("Data Source", value="GENSIS_forecast_model")
        model_version = st.text_input("Model Version", value="v2.1")
        
        # Generate button
        generate_btn = st.button("🎯 Generate Policy Recommendations", type="primary", use_container_width=True)
    
    # Main content area
    if generate_btn:
        # Prepare input data
        input_data = {
            "station_id": station_id,
            "location": {
                "city": city,
                "state": state,
                "country": country,
                "coordinates": {"lat": latitude, "lon": longitude}
            },
            "forecast_summary": {
                "predicted_aqi_value": aqi_value,
                "confidence": confidence,
                "time_horizon_hours": time_horizon
            },
            "drivers_summary": {
                "traffic": traffic,
                "dust": dust,
                "industry": industry,
                "weather_condition": weather,
                "seasonal_factor": seasonal
            },
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "data_source": data_source,
                "model_version": model_version,
                "notes": "Generated via Streamlit UI"
            }
        }
        
        # Generate recommendations
        with st.spinner("Generating policy recommendations..."):
            recommendations = st.session_state.policy_assistant.generate_policies(input_data)
        
        # Display results
        display_results(recommendations, input_data)
    else:
        # Default display
        st.info("👈 Configure the air quality parameters in the sidebar and click 'Generate Policy Recommendations' to see results.")
        
        # Show sample data visualization
        display_sample_dashboard()

def display_results(recommendations, input_data):
    # AQI Status Dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="AQI Value",
            value=recommendations["aqi"]["value"],
            delta=None
        )
    
    with col2:
        st.metric(
            label="Category", 
            value=recommendations["aqi"]["category"].replace("_", " "),
            delta=None
        )
    
    with col3:
        st.metric(
            label="Urgency Level",
            value=recommendations["aqi"]["urgency"].upper(),
            delta=None
        )
    
    with col4:
        st.metric(
            label="Total Policies",
            value=len(recommendations["policies"]),
            delta=None
        )
    
    # AQI Description
    st.markdown(f"""
    <div class="aqi-card">
        <h3>🌡️ Air Quality Status</h3>
        <p><strong>{recommendations["aqi"]["descriptor"]}</strong></p>
        <p><em>Station: {recommendations["station_id"]} | {input_data["location"]["city"]}, {input_data["location"]["country"]}</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pollution Drivers Summary
    st.subheader("🚗 Current Pollution Drivers")
    
    driver_cols = st.columns(5)
    drivers = recommendations["drivers"]
    driver_icons = {"traffic": "🚗", "dust": "🏗️", "industry": "🏭", "weather_condition": "🌤️", "seasonal_factor": "📅"}
    
    for i, (driver, value) in enumerate(drivers.items()):
        with driver_cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">{driver_icons.get(driver, "📊")}</div>
                <div><strong>{driver.replace('_', ' ').title()}</strong></div>
                <div style="color: #667eea; font-weight: bold;">{value.replace('_', ' ').title()}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Policy Recommendations by Category
    st.subheader("📋 Policy Recommendations")
    
    # Group policies by category
    policies_by_category = {}
    for policy in recommendations["policies"]:
        category = policy["category"]
        if category not in policies_by_category:
            policies_by_category[category] = []
        policies_by_category[category].append(policy)
    
    # Display policies in tabs
    if policies_by_category:
        category_icons = {
            "traffic": "🚗 Traffic Management",
            "dust": "🏗️ Dust Control", 
            "industry": "🏭 Industrial Controls",
            "public_health": "🏥 Public Health",
            "general": "🏛️ General Measures"
        }
        
        tabs = st.tabs([category_icons.get(cat, cat.title()) for cat in policies_by_category.keys()])
        
        for i, (category, policies) in enumerate(policies_by_category.items()):
            with tabs[i]:
                for j, policy in enumerate(policies):
                    with st.expander(f"Policy {j+1}: {policy['intervention']}", expanded=True):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Intervention:** {policy['intervention']}")
                            st.write(f"**Expected Impact:** {policy['expected_impact']['qualitative']}")
                            
                            if policy['expected_impact']['quantitative']['value'] > 0:
                                st.write(f"**Quantitative Impact:** {policy['expected_impact']['quantitative']['value']} {policy['expected_impact']['quantitative']['unit']}")
                            else:
                                st.write(f"**Impact Type:** {policy['expected_impact']['quantitative']['unit']}")
                            
                            st.write(f"**Rationale:** {policy['rationale']}")
                        
                        with col2:
                            urgency_class = f"urgency-{policy['urgency']}"
                            st.markdown(f"""
                            <div class="{urgency_class}" style="text-align: center; margin: 1rem 0;">
                                <strong>{policy['urgency'].upper()}</strong><br>
                                <small>PRIORITY</small>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Export Options
    st.subheader("📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as JSON"):
            json_str = json.dumps(recommendations, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"policy_recommendations_{recommendations['station_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Export as CSV"):
            # Convert policies to DataFrame
            policies_df = pd.DataFrame(recommendations["policies"])
            csv = policies_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"policy_recommendations_{recommendations['station_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📈 Generate Report"):
            generate_detailed_report(recommendations, input_data)

def display_sample_dashboard():
    st.subheader("📊 Sample Air Quality Dashboard")
    
    # Generate sample time series data
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
    sample_aqi = np.random.normal(150, 50, len(dates))
    sample_aqi = np.clip(sample_aqi, 0, 500)
    
    # AQI Trend Chart
    fig_aqi = go.Figure()
    fig_aqi.add_trace(go.Scatter(
        x=dates,
        y=sample_aqi,
        mode='lines',
        name='AQI',
        line=dict(color='#667eea', width=2)
    ))
    
    fig_aqi.update_layout(
        title="7-Day AQI Trend",
        xaxis_title="Date",
        yaxis_title="AQI Value",
        height=400
    )
    
    st.plotly_chart(fig_aqi, use_container_width=True)
    
    # Pollution Sources Chart
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for pollution sources
        sources = ['Traffic', 'Industry', 'Dust', 'Other']
        values = [35, 25, 20, 20]
        
        fig_pie = px.pie(
            values=values,
            names=sources,
            title="Pollution Source Contribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart for driver levels
        drivers = ['Traffic', 'Dust', 'Industry', 'Weather Impact']
        levels = [3, 3, 2, 4]  # 1=Low, 2=Medium, 3=High, 4=Critical
        
        fig_bar = px.bar(
            x=drivers,
            y=levels,
            title="Current Driver Levels",
            color=levels,
            color_continuous_scale='Reds'
        )
        fig_bar.update_layout(yaxis_title="Impact Level")
        st.plotly_chart(fig_bar, use_container_width=True)

def generate_detailed_report(recommendations, input_data):
    st.subheader("📋 Detailed Policy Report")
    
    report_content = f"""
    # Urban Air Quality Policy Report
    
    **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    **Station:** {recommendations['station_id']}
    **Location:** {input_data['location']['city']}, {input_data['location']['state']}, {input_data['location']['country']}
    
    ## Air Quality Status
    - **AQI Value:** {recommendations['aqi']['value']}
    - **Category:** {recommendations['aqi']['category'].replace('_', ' ')}
    - **Urgency Level:** {recommendations['aqi']['urgency'].upper()}
    - **Description:** {recommendations['aqi']['descriptor']}
    
    ## Current Pollution Drivers
    """
    
    for driver, value in recommendations['drivers'].items():
        report_content += f"- **{driver.replace('_', ' ').title()}:** {value.replace('_', ' ').title()}\n"
    
    report_content += f"""
    
    ## Policy Recommendations ({len(recommendations['policies'])} total)
    """
    
    for i, policy in enumerate(recommendations['policies'], 1):
        report_content += f"""
    ### {i}. {policy['intervention']} ({policy['category'].title()})
    - **Priority:** {policy['urgency'].upper()}
    - **Expected Impact:** {policy['expected_impact']['qualitative']}
    - **Quantitative Impact:** {policy['expected_impact']['quantitative']['value']} {policy['expected_impact']['quantitative']['unit']}
    - **Rationale:** {policy['rationale']}
    """
    
    st.markdown(report_content)
    
    # Download button for the report
    st.download_button(
        label="📄 Download Full Report",
        data=report_content,
        file_name=f"air_quality_policy_report_{recommendations['station_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

if __name__ == "__main__":
    main()