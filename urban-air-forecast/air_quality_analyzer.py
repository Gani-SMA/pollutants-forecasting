import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime
import requests
from io import BytesIO
import json
import base64

# Page configuration
st.set_page_config(
    page_title="📸 Air Quality Image Analyzer",
    page_icon="📸",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .aqi-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        color: white;
        font-weight: bold;
    }
    
    .aqi-good { background: linear-gradient(135deg, #48bb78, #38a169); }
    .aqi-moderate { background: linear-gradient(135deg, #ed8936, #dd6b20); }
    .aqi-unhealthy { background: linear-gradient(135deg, #e53e3e, #c53030); }
    .aqi-hazardous { background: linear-gradient(135deg, #742a2a, #63171b); }
    
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-desc {
        font-size: 0.9rem;
        color: #718096;
    }
    
    .recommendation-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .recommendation-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #2d3748;
        margin-bottom: 1rem;
    }
    
    .recommendation-item {
        background: #f7fafc;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    
    .success-msg {
        background: #c6f6d5;
        color: #22543d;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .error-msg {
        background: #fed7d7;
        color: #742a2a;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .info-msg {
        background: #bee3f8;
        color: #2a4365;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AirQualityAnalyzer:
    def __init__(self):
        self.aqi_categories = {
            'Good': {'range': (0, 50), 'class': 'aqi-good', 'color': '#48bb78'},
            'Moderate': {'range': (51, 100), 'class': 'aqi-moderate', 'color': '#ed8936'},
            'Unhealthy': {'range': (101, 200), 'class': 'aqi-unhealthy', 'color': '#e53e3e'},
            'Hazardous': {'range': (201, 500), 'class': 'aqi-hazardous', 'color': '#742a2a'}
        }
    
    def analyze_image(self, image):
        """Analyze image for air quality indicators"""
        try:
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Basic validation
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                return {'success': False, 'error': 'Invalid image format'}
            
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # 1. Visibility Analysis
            contrast = np.std(gray) / 255.0
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            visibility_score = min(1.0, (contrast * 0.7 + edge_density * 0.3) * 2)
            
            # 2. Haze Analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            avg_saturation = np.mean(saturation) / 255.0
            haze_density = max(0, min(1, 1.0 - avg_saturation))
            
            # 3. Sky Color Analysis
            height, width = img_array.shape[:2]
            sky_region = img_array[:height//3, :, :]
            avg_r = np.mean(sky_region[:, :, 0])
            avg_g = np.mean(sky_region[:, :, 1])
            avg_b = np.mean(sky_region[:, :, 2])
            
            # Calculate pollution tint
            total_rgb = avg_r + avg_g + avg_b + 1
            pollution_tint = max(0, (avg_r + avg_g - avg_b) / total_rgb)
            
            # 4. Particulate Detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_variance = np.var(laplacian)
            particulate_score = min(1.0, noise_variance / 8000)
            
            # 5. Calculate AQI
            pollution_index = (
                (1 - visibility_score) * 0.25 +
                haze_density * 0.30 +
                pollution_tint * 0.20 +
                particulate_score * 0.25
            )
            
            estimated_aqi = int(max(10, min(500, pollution_index * 400)))
            category = self._get_category(estimated_aqi)
            
            return {
                'success': True,
                'aqi': estimated_aqi,
                'category': category,
                'visibility_score': round(visibility_score, 3),
                'haze_density': round(haze_density, 3),
                'pollution_tint': round(pollution_tint, 3),
                'particulate_score': round(particulate_score, 3),
                'sky_rgb': [int(avg_r), int(avg_g), int(avg_b)],
                'pollution_index': round(pollution_index, 3),
                'analysis_time': datetime.now().strftime('%H:%M:%S')
            }
            
        except Exception as e:
            return {
                'success': False, 
                'error': f'Analysis failed: {str(e)}'
            }
    
    def _get_category(self, aqi):
        for category, info in self.aqi_categories.items():
            if info['range'][0] <= aqi <= info['range'][1]:
                return category
        return 'Hazardous'
    
    def get_health_recommendations(self, aqi):
        """Get specific health recommendations based on AQI level"""
        if aqi <= 50:  # Good
            return {
                'immediate': [
                    f"Air quality is excellent (AQI: {aqi}). No health concerns.",
                    "Perfect conditions for all outdoor activities.",
                    "Ideal time for exercise, sports, and outdoor recreation."
                ],
                'protective': [
                    "No protective measures needed.",
                    "Consider opening windows for natural ventilation.",
                    "Great opportunity for outdoor activities with family."
                ],
                'activities': [
                    "All outdoor exercises and sports are safe.",
                    "Children can play outside without restrictions.",
                    "Perfect for hiking, cycling, and outdoor events."
                ]
            }
        elif aqi <= 100:  # Moderate
            return {
                'immediate': [
                    f"Air quality is acceptable (AQI: {aqi}) for most people.",
                    "Sensitive individuals may experience minor irritation.",
                    "General population can continue normal activities."
                ],
                'protective': [
                    "Sensitive people should consider wearing masks outdoors.",
                    "Use air purifiers indoors if available.",
                    "Keep rescue medications handy if you have asthma."
                ],
                'activities': [
                    "Limit prolonged outdoor exertion for sensitive groups.",
                    "Monitor children and elderly for respiratory symptoms.",
                    "Reduce intensity of outdoor sports activities."
                ]
            }
        elif aqi <= 200:  # Unhealthy
            return {
                'immediate': [
                    f"Unhealthy air quality (AQI: {aqi}). Everyone at risk.",
                    "Stay indoors as much as possible.",
                    "Health effects likely for sensitive groups."
                ],
                'protective': [
                    "Wear N95 or KN95 masks when going outside.",
                    "Use air purifiers and keep windows closed.",
                    "Stay hydrated and avoid strenuous activities."
                ],
                'activities': [
                    "Cancel outdoor exercise and sports events.",
                    "Keep children indoors during peak pollution hours.",
                    "Schools should move activities indoors."
                ]
            }
        else:  # Hazardous
            return {
                'immediate': [
                    f"HAZARDOUS air quality (AQI: {aqi}). Health emergency!",
                    "Avoid ALL outdoor exposure immediately.",
                    "Seek medical attention if experiencing symptoms."
                ],
                'protective': [
                    "Use P100 respirators if outdoor exposure unavoidable.",
                    "Seal windows/doors, use multiple air purifiers.",
                    "Have emergency medications readily accessible."
                ],
                'activities': [
                    "Schools and offices should close immediately.",
                    "Avoid all non-essential travel and outdoor work.",
                    "Emergency services only for critical situations."
                ]
            }

def display_results(results, analyzer):
    """Display analysis results using native Streamlit components"""
    if not results['success']:
        st.error(f"❌ {results['error']}")
        return
    
    # AQI Display with color coding
    aqi_color = analyzer.aqi_categories[results['category']]['color']
    
    # Create AQI display using metrics
    col_aqi1, col_aqi2, col_aqi3 = st.columns([1, 2, 1])
    with col_aqi2:
        st.metric(
            label="🌍 Air Quality Index (AQI)",
            value=f"{results['aqi']} - {results['category']}",
            help=f"Analysis completed at {results['analysis_time']}"
        )
    
    # Analysis Details using native components
    st.subheader("🔍 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Visibility
        visibility_pct = int(results['visibility_score'] * 100)
        st.metric(
            label="👁️ Atmospheric Visibility",
            value=f"{visibility_pct}%",
            help="Higher percentage indicates clearer air"
        )
        
        # Haze
        haze_pct = int(results['haze_density'] * 100)
        st.metric(
            label="🌫️ Haze Level",
            value=f"{haze_pct}%",
            help="Higher percentage indicates more haze/smog"
        )
    
    with col2:
        # Sky Color
        r, g, b = results['sky_rgb']
        st.metric(
            label="🌤️ Sky Color Analysis",
            value=f"RGB({r}, {g}, {b})",
            help=f"Pollution Tint: {results['pollution_tint']:.3f}"
        )
        
        # Particulates
        particulate_pct = int(results['particulate_score'] * 100)
        st.metric(
            label="💨 Particulate Matter",
            value=f"{particulate_pct}%",
            help="Visible particles and dust in air"
        )

def display_recommendations(recommendations):
    """Display health recommendations using native Streamlit components"""
    st.subheader("🏥 Health Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🚨 Immediate Actions")
        for i, action in enumerate(recommendations['immediate'], 1):
            st.info(f"**{i}.** {action}")
    
    with col2:
        st.markdown("### 🛡️ Protective Measures")
        for i, measure in enumerate(recommendations['protective'], 1):
            st.success(f"**{i}.** {measure}")
    
    with col3:
        st.markdown("### 🏃‍♂️ Activity Guidelines")
        for i, activity in enumerate(recommendations['activities'], 1):
            st.warning(f"**{i}.** {activity}")

def main():
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = AirQualityAnalyzer()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📸 Air Quality Image Analyzer</h1>
        <p>Upload or capture images to analyze air quality and get instant health recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📸 Image Input")
        
        input_method = st.radio(
            "Choose input method:",
            ["📁 Upload Image", "📷 Camera Capture", "🌐 URL Input"]
        )
        
        # Settings
        st.header("⚙️ Settings")
        auto_analyze = st.checkbox("Auto-analyze on upload", value=True)
        show_tips = st.checkbox("Show analysis tips", value=True)
        
        uploaded_image = None
        
        # Image input handling
        if input_method == "📁 Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                help="Upload an outdoor image for best results"
            )
            if uploaded_file:
                uploaded_image = Image.open(uploaded_file)
                st.success("✅ Image uploaded successfully!")
                
        elif input_method == "📷 Camera Capture":
            camera_image = st.camera_input("Take a picture of outdoor environment")
            if camera_image:
                uploaded_image = Image.open(camera_image)
                st.success("📸 Photo captured successfully!")
                
        elif input_method == "🌐 URL Input":
            url = st.text_input("Enter image URL:")
            if url:
                try:
                    response = requests.get(url, timeout=10)
                    uploaded_image = Image.open(BytesIO(response.content))
                    st.success("✅ Image loaded from URL!")
                except Exception as e:
                    st.error(f"❌ Failed to load image: {str(e)}")
        
        # Tips
        if show_tips:
            st.markdown("""
            ### 💡 Tips for Best Results
            - Use **outdoor images** with visible sky
            - Include **distant objects** for visibility analysis
            - Ensure **good lighting** conditions
            - Avoid extreme close-ups or indoor scenes
            """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Image Preview")
        
        if uploaded_image:
            st.image(uploaded_image, width='stretch', caption="Uploaded Image")
            
            # Analysis trigger
            should_analyze = False
            
            if auto_analyze:
                should_analyze = True
                st.markdown('<div class="info-msg">🔄 Auto-analyzing image...</div>', unsafe_allow_html=True)
            else:
                if st.button("🔬 Analyze Air Quality", type="primary"):
                    should_analyze = True
            
            # Perform analysis
            if should_analyze:
                with st.spinner("🔍 Analyzing air quality indicators..."):
                    try:
                        results = st.session_state.analyzer.analyze_image(uploaded_image)
                        st.session_state.results = results
                        
                        if results['success']:
                            st.success("✅ Analysis completed successfully!")
                            # Force rerun to show recommendations immediately
                            st.rerun()
                        else:
                            st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error during analysis: {str(e)}")
                        st.session_state.results = {'success': False, 'error': str(e)}
        else:
            st.markdown("""
            <div class="info-msg">
                <h4>📋 Instructions</h4>
                <p>1. Choose an input method from the sidebar</p>
                <p>2. Upload an image, capture a photo, or enter a URL</p>
                <p>3. The analysis will start automatically (or click Analyze button)</p>
                <p>4. View results and health recommendations below</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if 'results' in st.session_state:
            display_results(st.session_state.results, st.session_state.analyzer)
        else:
            st.markdown("""
            <div class="info-msg">
                <h4>📊 Results will appear here</h4>
                <p><strong>AQI Estimate:</strong> 0-500 scale air quality index</p>
                <p><strong>Category:</strong> Good, Moderate, Unhealthy, or Hazardous</p>
                <p><strong>Detailed Metrics:</strong> Visibility, haze, sky color, particulates</p>
                <p><strong>Health Advice:</strong> Personalized recommendations</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Health Recommendations Section
    if 'results' in st.session_state and st.session_state.results['success']:
        st.markdown("---")  # Separator
        
        # Real-time status alert
        aqi = st.session_state.results['aqi']
        category = st.session_state.results['category']
        
        if aqi <= 50:
            st.success(f"🟢 **GOOD AIR QUALITY** - AQI {aqi} ({category})")
        elif aqi <= 100:
            st.warning(f"🟡 **MODERATE AIR QUALITY** - AQI {aqi} ({category})")
        elif aqi <= 200:
            st.error(f"🔴 **UNHEALTHY AIR QUALITY** - AQI {aqi} ({category})")
        else:
            st.error(f"🚨 **HAZARDOUS AIR QUALITY** - AQI {aqi} ({category})")
        
        # Get and display recommendations
        recommendations = st.session_state.analyzer.get_health_recommendations(aqi)
        display_recommendations(recommendations)
        
        # Export Section
        st.header("📤 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download JSON Report"):
                report_data = {
                    'timestamp': datetime.now().isoformat(),
                    'analysis_results': st.session_state.results,
                    'health_recommendations': recommendations
                }
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"air_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📝 Download Summary"):
                summary = f"""# Air Quality Analysis Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**AQI:** {st.session_state.results['aqi']} ({st.session_state.results['category']})

## Analysis Details
- Visibility Score: {st.session_state.results['visibility_score']:.3f}
- Haze Density: {st.session_state.results['haze_density']:.3f}
- Pollution Tint: {st.session_state.results['pollution_tint']:.3f}
- Particulate Score: {st.session_state.results['particulate_score']:.3f}

## Health Recommendations
### Immediate Actions
{chr(10).join(['- ' + action for action in recommendations['immediate']])}

### Protective Measures
{chr(10).join(['- ' + measure for measure in recommendations['protective']])}

### Activity Guidelines
{chr(10).join(['- ' + activity for activity in recommendations['activities']])}
"""
                
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name=f"air_quality_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        with col3:
            if st.button("📊 Download CSV Data", use_container_width=True):
                import pandas as pd
                
                data = {
                    'Metric': ['AQI', 'Category', 'Visibility', 'Haze', 'Pollution Tint', 'Particulates'],
                    'Value': [
                        st.session_state.results['aqi'],
                        st.session_state.results['category'],
                        st.session_state.results['visibility_score'],
                        st.session_state.results['haze_density'],
                        st.session_state.results['pollution_tint'],
                        st.session_state.results['particulate_score']
                    ]
                }
                
                df = pd.DataFrame(data)
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"air_quality_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()