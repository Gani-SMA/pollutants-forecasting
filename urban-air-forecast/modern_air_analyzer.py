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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to match the exact UI design
st.markdown("""
<style>
    /* Dark theme styling */
    .stApp {
        background-color: #1e1e1e;
        color: white;
    }
    
    /* Main header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Section headers */
    .section-header {
        color: #e2e8f0;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* AQI Card */
    .aqi-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        color: white;
        font-weight: bold;
    }
    
    .aqi-good { background: linear-gradient(135deg, #48bb78, #38a169); }
    .aqi-moderate { background: linear-gradient(135deg, #ed8936, #dd6b20); }
    .aqi-unhealthy { background: linear-gradient(135deg, #e53e3e, #c53030); }
    .aqi-hazardous { background: linear-gradient(135deg, #742a2a, #63171b); }
    
    .aqi-number {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
    }
    
    .aqi-category {
        font-size: 1.5rem;
        margin: 0.5rem 0;
    }
    
    .aqi-time {
        font-size: 0.9rem;
        opacity: 0.8;
        margin: 0;
    }
    
    /* Analysis cards */
    .analysis-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .metric-icon {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-title {
        color: #e2e8f0;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-desc {
        color: #a0aec0;
        font-size: 0.8rem;
        margin: 0;
    }
    
    /* Color coding for metrics */
    .visibility-good { color: #48bb78; }
    .visibility-moderate { color: #ed8936; }
    .visibility-poor { color: #e53e3e; }
    
    .haze-low { color: #48bb78; }
    .haze-moderate { color: #ed8936; }
    .haze-high { color: #e53e3e; }
    
    .particulate-low { color: #48bb78; }
    .particulate-moderate { color: #ed8936; }
    .particulate-high { color: #e53e3e; }
    
    /* Health recommendations */
    .health-section {
        margin: 2rem 0;
    }
    
    .recommendation-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .recommendation-card {
        background: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .recommendation-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
        color: #e2e8f0;
    }
    
    .recommendation-item {
        background: #1a202c;
        border-left: 3px solid #667eea;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #e2e8f0;
    }
    
    /* Export section */
    .export-section {
        margin: 2rem 0;
    }
    
    .export-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .export-button {
        background: #4a5568;
        border: 1px solid #718096;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        color: #e2e8f0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .export-button:hover {
        background: #718096;
        transform: translateY(-2px);
    }
    
    /* Status messages */
    .status-analyzing {
        background: #3182ce;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .status-success {
        background: #38a169;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .status-error {
        background: #e53e3e;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #2d3748;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #4a5568;
    }
    
    .sidebar-header {
        color: #e2e8f0;
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Tips section */
    .tips-section {
        background: #2a4365;
        border-left: 4px solid #3182ce;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .tips-header {
        color: #90cdf4;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .tips-list {
        color: #bee3f8;
        font-size: 0.9rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

class ModernAirQualityAnalyzer:
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
    
    def get_health_recommendations(self, aqi, analysis_data=None):
        """Get comprehensive health recommendations based on AQI and specific parameters"""
        
        # Base recommendations by AQI level
        base_recommendations = self._get_base_recommendations(aqi)
        
        # Add parameter-specific recommendations if analysis data is available
        if analysis_data:
            parameter_specific = self._get_parameter_specific_recommendations(analysis_data)
            
            # Merge recommendations
            for category in base_recommendations:
                base_recommendations[category].extend(parameter_specific.get(category, []))
        
        return base_recommendations
    
    def _get_base_recommendations(self, aqi):
        """Get base health recommendations based on AQI"""
        if aqi <= 50:  # Good
            return {
                'immediate': [
                    "🌟 Air quality is excellent for all outdoor activities",
                    "✅ No health precautions necessary",
                    "🏃‍♂️ Perfect conditions for exercise and sports",
                    "👶 Safe for children to play outside"
                ],
                'protective': [
                    "🌱 Consider adding air-purifying plants indoors",
                    "🪟 Keep windows open for natural ventilation",
                    "🚴‍♂️ Great day for cycling and outdoor activities",
                    "🌳 Enjoy parks and outdoor recreational areas"
                ],
                'activities': [
                    "🏃‍♂️ All outdoor exercises are completely safe",
                    "👶 Children can engage in all outdoor activities",
                    "🏫 Normal school outdoor activities and sports",
                    "🚶‍♀️ Walking and jogging are highly recommended"
                ]
            }
        elif aqi <= 100:  # Moderate
            return {
                'immediate': [
                    "⚠️ Air quality is acceptable for most people",
                    "🔍 Sensitive individuals may experience minor issues",
                    "✅ General population can continue normal activities",
                    "👀 Monitor air quality if you have respiratory conditions"
                ],
                'protective': [
                    "😷 Sensitive individuals consider wearing masks outdoors",
                    "🏠 Use air purifiers if available indoors",
                    "💊 Keep rescue medications handy for respiratory conditions",
                    "🌬️ Limit time in high-traffic areas"
                ],
                'activities': [
                    "🏃‍♂️ Limit prolonged outdoor exertion for sensitive groups",
                    "👶 Monitor children for any respiratory symptoms",
                    "🏫 Reduce intensity of outdoor sports activities",
                    "🚶‍♀️ Short walks are generally safe for most people"
                ]
            }
        elif aqi <= 200:  # Unhealthy
            return {
                'immediate': [
                    "🚨 Everyone should limit outdoor activities",
                    "🏠 Stay indoors when possible, especially vulnerable groups",
                    "⚠️ Health effects may be experienced by everyone",
                    "🔴 Avoid prolonged outdoor exposure"
                ],
                'protective': [
                    "😷 Wear N95 or KN95 masks when going outside",
                    "🏠 Use air purifiers and keep windows closed",
                    "💧 Stay well hydrated and avoid strenuous activities",
                    "🚗 Use air recirculation mode in vehicles"
                ],
                'activities': [
                    "🏃‍♂️ Cancel outdoor exercise and sports events",
                    "👶 Keep children indoors as much as possible",
                    "🏫 Schools should limit or cancel outdoor activities",
                    "🚶‍♀️ Postpone non-essential outdoor activities"
                ]
            }
        else:  # Hazardous
            return {
                'immediate': [
                    "🚨 HEALTH EMERGENCY: Avoid all outdoor exposure",
                    "🏥 Seek medical attention if experiencing symptoms",
                    "🏠 Stay indoors with air filtration systems running",
                    "📞 Contact healthcare provider if you have respiratory issues"
                ],
                'protective': [
                    "😷 Use P100 respirators if outdoor exposure unavoidable",
                    "🏠 Seal windows and doors, use multiple air purifiers",
                    "💊 Have emergency medications readily accessible",
                    "🚗 Avoid driving unless absolutely necessary"
                ],
                'activities': [
                    "🏫 Schools and offices should close or go remote",
                    "🚗 Avoid all non-essential travel and outdoor work",
                    "🏥 Emergency services only for critical needs",
                    "📱 Stay informed about air quality updates"
                ]
            }
    
    def _get_parameter_specific_recommendations(self, analysis_data):
        """Get specific recommendations based on individual parameters"""
        recommendations = {'immediate': [], 'protective': [], 'activities': []}
        
        # Visibility-specific recommendations
        visibility = analysis_data.get('visibility_score', 0)
        if visibility < 0.3:  # Very poor visibility
            recommendations['immediate'].append("👁️ Extremely poor visibility detected - avoid driving if possible")
            recommendations['protective'].append("🚗 Use headlights and drive slowly due to reduced visibility")
            recommendations['activities'].append("🚶‍♀️ Avoid outdoor navigation activities")
        elif visibility < 0.5:  # Poor visibility
            recommendations['immediate'].append("👁️ Reduced visibility detected - exercise caution outdoors")
            recommendations['protective'].append("🔦 Carry flashlight for outdoor activities")
        
        # Haze-specific recommendations
        haze = analysis_data.get('haze_density', 0)
        if haze > 0.7:  # High haze
            recommendations['immediate'].append("🌫️ Heavy haze detected - may cause eye and throat irritation")
            recommendations['protective'].append("👓 Wear protective eyewear to prevent eye irritation")
            recommendations['activities'].append("🏃‍♂️ Avoid intense outdoor workouts due to haze")
        elif haze > 0.5:  # Moderate haze
            recommendations['immediate'].append("🌫️ Moderate haze present - sensitive individuals take precautions")
            recommendations['protective'].append("💧 Keep eyes moist with artificial tears if needed")
        
        # Particulate-specific recommendations
        particulates = analysis_data.get('particulate_score', 0)
        if particulates > 0.6:  # High particulates
            recommendations['immediate'].append("💨 High particulate matter detected - respiratory protection advised")
            recommendations['protective'].append("😷 N95 masks strongly recommended for all outdoor activities")
            recommendations['activities'].append("🏠 Consider indoor alternatives for exercise")
        elif particulates > 0.4:  # Moderate particulates
            recommendations['immediate'].append("💨 Elevated particulates - monitor respiratory symptoms")
            recommendations['protective'].append("🌬️ Avoid areas with heavy traffic or construction")
        
        # Pollution tint specific recommendations
        pollution_tint = analysis_data.get('pollution_tint', 0)
        if pollution_tint > 0.3:  # High pollution tint
            recommendations['immediate'].append("🌤️ Significant atmospheric pollution detected in sky analysis")
            recommendations['protective'].append("🏠 Keep windows closed and use air filtration")
            recommendations['activities'].append("🌳 Seek areas with more vegetation for cleaner air")
        
        # Combined parameter warnings
        if visibility < 0.4 and haze > 0.6:
            recommendations['immediate'].append("⚠️ Poor visibility + heavy haze: Hazardous conditions for travel")
        
        if particulates > 0.5 and pollution_tint > 0.3:
            recommendations['protective'].append("🚨 High pollution levels: Consider postponing outdoor activities")
        
        return recommendations

def get_metric_color_class(value, metric_type):
    """Get color class based on metric value and type"""
    if metric_type == 'visibility':
        if value >= 0.7:
            return 'visibility-good'
        elif value >= 0.4:
            return 'visibility-moderate'
        else:
            return 'visibility-poor'
    elif metric_type == 'haze':
        if value <= 0.3:
            return 'haze-low'
        elif value <= 0.6:
            return 'haze-moderate'
        else:
            return 'haze-high'
    elif metric_type == 'particulate':
        if value <= 0.3:
            return 'particulate-low'
        elif value <= 0.6:
            return 'particulate-moderate'
        else:
            return 'particulate-high'
    return ''

def main():
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ModernAirQualityAnalyzer()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📸 Air Quality Image Analyzer</h1>
        <p>Upload or capture images to analyze air quality and get instant health recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📸 Image Input</div>', unsafe_allow_html=True)
        
        input_method = st.radio(
            "Choose input method:",
            ["📁 Upload Image", "📷 Camera Capture", "🌐 URL Input"],
            key="input_method"
        )
        
        # Settings
        st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)
        auto_analyze = st.checkbox("Auto-analyze on upload", value=True)
        show_analysis_tips = st.checkbox("Show analysis tips", value=True)
        
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
                st.success("📸 Photo captured successfully!")
                
        elif input_method == "📷 Camera Capture":
            st.markdown("Take a picture of outdoor environment")
            camera_image = st.camera_input("Take picture")
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
        
        # Tips section
        if show_analysis_tips:
            st.markdown("""
            <div class="tips-section">
                <div class="tips-header">💡 Tips for Best Results</div>
                <div class="tips-list">
                    • Use <strong>outdoor images</strong> with visible sky<br>
                    • Include <strong>distant objects</strong> for visibility analysis<br>
                    • Ensure <strong>good lighting</strong> conditions<br>
                    • Avoid extreme close-ups or indoor scenes
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">📸 Image Preview</div>', unsafe_allow_html=True)
        
        if uploaded_image:
            st.image(uploaded_image, width='stretch', caption="Uploaded Image")
            
            # Analysis trigger
            should_analyze = False
            
            if auto_analyze:
                should_analyze = True
                st.markdown('<div class="status-analyzing">🔄 Auto-analyzing image...</div>', unsafe_allow_html=True)
            else:
                if st.button("🔬 Analyze Air Quality", type="primary"):
                    should_analyze = True
            
            # Perform analysis
            if should_analyze:
                with st.spinner("🔍 Analyzing air quality indicators..."):
                    results = st.session_state.analyzer.analyze_image(uploaded_image)
                    st.session_state.results = results
                    
                    if results['success']:
                        st.markdown('<div class="status-success">✅ Analysis completed successfully!</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-error">❌ Analysis failed: {results.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Please upload an image, capture a photo, or enter a URL to start analysis")
    
    with col2:
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        if 'results' in st.session_state and st.session_state.results['success']:
            results = st.session_state.results
            
            # AQI Display
            aqi_class = st.session_state.analyzer.aqi_categories[results['category']]['class']
            st.markdown(f"""
            <div class="aqi-card {aqi_class}">
                <div class="aqi-number">AQI: {results['aqi']}</div>
                <div class="aqi-category">{results['category']}</div>
                <div class="aqi-time">Analysis completed at {results['analysis_time']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed Analysis
            st.markdown('<div class="section-header">🔍 Detailed Analysis</div>', unsafe_allow_html=True)
            
            # Create 2x2 grid for metrics
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Atmospheric Visibility
                visibility_pct = int(results['visibility_score'] * 100)
                visibility_class = get_metric_color_class(results['visibility_score'], 'visibility')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">👁️</div>
                    <div class="metric-title">Atmospheric Visibility</div>
                    <div class="metric-value {visibility_class}">{visibility_pct}%</div>
                    <div class="metric-desc">Higher percentage indicates clearer air</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Haze Level
                haze_pct = int(results['haze_density'] * 100)
                haze_class = get_metric_color_class(results['haze_density'], 'haze')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">🌫️</div>
                    <div class="metric-title">Haze Level</div>
                    <div class="metric-value {haze_class}">{haze_pct}%</div>
                    <div class="metric-desc">Higher percentage indicates more haze/smog</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                # Sky Color Analysis
                r, g, b = results['sky_rgb']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">🌤️</div>
                    <div class="metric-title">Sky Color Analysis</div>
                    <div class="metric-value" style="color: #667eea;">RGB({r}, {g}, {b})</div>
                    <div class="metric-desc">Pollution Tint: {results['pollution_tint']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Particulate Matter
                particulate_pct = int(results['particulate_score'] * 100)
                particulate_class = get_metric_color_class(results['particulate_score'], 'particulate')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">💨</div>
                    <div class="metric-title">Particulate Matter</div>
                    <div class="metric-value {particulate_class}">{particulate_pct}%</div>
                    <div class="metric-desc">Visible particles and dust in air</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📊 Analysis results will appear here after processing an image")
    
    # Health Recommendations Section
    if 'results' in st.session_state and st.session_state.results['success']:
        st.markdown('<div class="section-header">🏥 Health Recommendations</div>', unsafe_allow_html=True)
        
        recommendations = st.session_state.analyzer.get_health_recommendations(
            st.session_state.results['aqi'], 
            st.session_state.results
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="recommendation-card">
                <div class="recommendation-header">🚨 Immediate Actions</div>
                {''.join([f'<div class="recommendation-item">{item}</div>' for item in recommendations['immediate']])}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="recommendation-card">
                <div class="recommendation-header">🛡️ Protective Measures</div>
                {''.join([f'<div class="recommendation-item">{item}</div>' for item in recommendations['protective']])}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="recommendation-card">
                <div class="recommendation-header">🏃‍♂️ Activity Guidelines</div>
                {''.join([f'<div class="recommendation-item">{item}</div>' for item in recommendations['activities']])}
            </div>
            """, unsafe_allow_html=True)
        
        # Export Section
        st.markdown('<div class="section-header">📤 Export Results</div>', unsafe_allow_html=True)
        
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
- Atmospheric Visibility: {int(st.session_state.results['visibility_score'] * 100)}%
- Haze Level: {int(st.session_state.results['haze_density'] * 100)}%
- Sky Color: RGB{st.session_state.results['sky_rgb']}
- Particulate Matter: {int(st.session_state.results['particulate_score'] * 100)}%

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
                    mime="text/markdown",
                    use_container_width=True
                )
        
        with col3:
            if st.button("📊 Download CSV Data"):
                import pandas as pd
                
                data = {
                    'Metric': ['AQI', 'Category', 'Visibility (%)', 'Haze (%)', 'Pollution Tint', 'Particulates (%)'],
                    'Value': [
                        st.session_state.results['aqi'],
                        st.session_state.results['category'],
                        int(st.session_state.results['visibility_score'] * 100),
                        int(st.session_state.results['haze_density'] * 100),
                        st.session_state.results['pollution_tint'],
                        int(st.session_state.results['particulate_score'] * 100)
                    ]
                }
                
                df = pd.DataFrame(data)
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"air_quality_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()