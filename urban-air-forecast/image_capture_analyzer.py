import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
from io import BytesIO
import base64
import json
import time
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="📸 Air Quality Image Analyzer",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .analysis-section {
        background: #1e1e2e;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .aqi-display {
        background: linear-gradient(135deg, #dc2626, #991b1b);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(220, 38, 38, 0.3);
    }
    
    .aqi-display h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 800;
    }
    
    .aqi-display h3 {
        font-size: 1.5rem;
        margin: 0.5rem 0;
        opacity: 0.9;
    }
    
    .detail-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .health-recommendation {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: #1a1a1a;
        border-left: 5px solid #3b82f6;
    }
    
    .immediate-action {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .protective-measure {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .activity-guideline {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .export-button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        margin: 0.5rem;
        cursor: pointer;
    }
    
    .status-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-analyzing {
        background: #3b82f6;
        color: white;
    }
    
    .status-complete {
        background: #10b981;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class ImageAirQualityAnalyzer:
    """Advanced image-based air quality analyzer with UI integration"""
    
    def __init__(self):
        self.aqi_categories = {
            'Good': {
                'range': (0, 50),
                'color': '#48bb78',
                'class': 'aqi-good'
            },
            'Moderate': {
                'range': (51, 100),
                'color': '#ed8936',
                'class': 'aqi-moderate'
            },
            'Unhealthy for Sensitive Groups': {
                'range': (101, 150),
                'color': '#f56565',
                'class': 'aqi-unhealthy-sensitive'
            },
            'Unhealthy': {
                'range': (151, 200),
                'color': '#9f7aea',
                'class': 'aqi-unhealthy'
            },
            'Very Unhealthy': {
                'range': (201, 300),
                'color': '#742a2a',
                'class': 'aqi-very-unhealthy'
            },
            'Hazardous': {
                'range': (301, 500),
                'color': '#742a2a',
                'class': 'aqi-hazardous'
            }
        }
    
    def analyze_visibility(self, image):
        """Analyze atmospheric visibility from image"""
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for visibility analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Contrast analysis
        contrast = np.std(gray) / 255.0
        
        # Visibility score (0-1, higher is better)
        visibility_score = min(1.0, (contrast * 0.7 + edge_density * 0.3) * 2)
        
        return {
            'visibility_score': visibility_score,
            'contrast': contrast,
            'edge_density': edge_density,
            'interpretation': self._interpret_visibility(visibility_score)
        }
    
    def analyze_haze_smog(self, image):
        """Analyze haze and smog levels"""
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Saturation analysis (haze reduces color saturation)
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation) / 255.0
        
        # Brightness uniformity (haze creates uniform brightness)
        value = hsv[:, :, 2]
        brightness_std = np.std(value) / 255.0
        
        # Haze density calculation
        haze_density = 1.0 - (avg_saturation * 0.6 + brightness_std * 0.4)
        haze_density = max(0, min(1, haze_density))
        
        return {
            'haze_density': haze_density,
            'avg_saturation': avg_saturation,
            'brightness_variation': brightness_std,
            'interpretation': self._interpret_haze(haze_density)
        }
    
    def analyze_sky_pollution(self, image):
        """Analyze sky color for pollution indicators"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Extract sky region (top third of image)
        sky_region = img_array[:height//3, :]
        
        # Average RGB values
        avg_r = np.mean(sky_region[:, :, 0])
        avg_g = np.mean(sky_region[:, :, 1])
        avg_b = np.mean(sky_region[:, :, 2])
        
        # Pollution tint (brown/yellow indicates pollution)
        pollution_tint = (avg_r + avg_g - avg_b) / (avg_r + avg_g + avg_b + 1)
        
        # Gray level (pollution desaturates colors)
        gray_level = 1 - (max(avg_r, avg_g, avg_b) - min(avg_r, avg_g, avg_b)) / 255.0
        
        return {
            'sky_rgb': [int(avg_r), int(avg_g), int(avg_b)],
            'pollution_tint': max(0, pollution_tint),
            'gray_level': gray_level,
            'interpretation': self._interpret_sky_color(pollution_tint, gray_level)
        }
    
    def detect_particulates(self, image):
        """Detect visible particles and dust"""
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Noise analysis for particulate detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_variance = np.var(laplacian)
        noise_score = min(1.0, noise_variance / 10000)
        
        # Texture analysis
        kernel = np.ones((5, 5), np.float32) / 25
        filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        texture = np.mean(np.abs(gray.astype(np.float32) - filtered)) / 255.0
        texture_score = min(1.0, texture / 5.0)
        
        # Particulate score
        particulate_score = (noise_score + texture_score) / 2
        
        return {
            'particulate_score': particulate_score,
            'texture': texture,
            'noise_variance': noise_variance,
            'interpretation': self._interpret_particulates(particulate_score)
        }
    
    def estimate_aqi_from_visual(self, visibility, haze, sky, particulates):
        """Estimate AQI from visual indicators"""
        # Weighted combination of pollution indicators
        pollution_index = (
            (1 - visibility['visibility_score']) * 0.25 +  # Poor visibility = pollution
            haze['haze_density'] * 0.30 +  # Haze indicates pollution
            sky['pollution_tint'] * 0.20 +  # Color tint from pollution
            particulates['particulate_score'] * 0.25  # Visible particles
        )
        
        # Convert to AQI scale (0-500)
        estimated_aqi = pollution_index * 350
        estimated_aqi = max(10, min(500, estimated_aqi))
        
        # Determine category
        category = self._get_aqi_category(estimated_aqi)
        
        # Calculate confidence based on indicator agreement
        indicators = [
            1 - visibility['visibility_score'],
            haze['haze_density'],
            sky['pollution_tint'],
            particulates['particulate_score']
        ]
        confidence = 1 - (np.std(indicators) / (np.mean(indicators) + 0.1))  # Avoid division by zero
        confidence = max(0.3, min(1.0, confidence))
        
        return {
            'estimated_aqi': int(estimated_aqi),
            'category': category,
            'pollution_index': pollution_index,
            'confidence': confidence,
            'color': self.aqi_categories[category]['color'],
            'class': self.aqi_categories[category]['class']
        }
    
    def _get_aqi_category(self, estimated_aqi):
        """Determine AQI category"""
        for category, info in self.aqi_categories.items():
            if info['range'][0] <= estimated_aqi <= info['range'][1]:
                return category
        return 'Hazardous'  # Default for values > 500
    
    def _interpret_visibility(self, score):
        if score > 0.8:
            return "Excellent visibility - Clear atmospheric conditions"
        elif score > 0.6:
            return "Good visibility - Minor atmospheric impact"
        elif score > 0.4:
            return "Moderate visibility - Noticeable air quality degradation"
        elif score > 0.2:
            return "Poor visibility - Significant pollution present"
        else:
            return "Very poor visibility - Hazardous air quality"
    
    def _interpret_haze(self, density):
        if density < 0.2:
            return "Clear atmosphere - No significant haze"
        elif density < 0.4:
            return "Light haze - Minor air quality impact"
        elif density < 0.6:
            return "Moderate haze - Noticeable air quality degradation"
        elif density < 0.8:
            return "Heavy haze - Poor air quality conditions"
        else:
            return "Dense haze/smog - Hazardous air quality"
    
    def _interpret_sky_color(self, tint, gray):
        if tint < 0.1 and gray < 0.3:
            return "Natural sky color - Clean air conditions"
        elif tint < 0.2 and gray < 0.5:
            return "Slight discoloration - Minor pollution"
        elif tint < 0.3 and gray < 0.7:
            return "Noticeable pollution tint - Moderate air quality impact"
        else:
            return "Strong pollution coloration - Poor air quality"
    
    def _interpret_particulates(self, score):
        if score < 0.2:
            return "No visible particulates - Clean air"
        elif score < 0.4:
            return "Minor particulate levels - Acceptable air quality"
        elif score < 0.6:
            return "Moderate particulate presence - Some concern"
        elif score < 0.8:
            return "High particulate levels - Poor air quality"
        else:
            return "Very high particulate levels - Hazardous conditions"

def get_dynamic_health_recommendations(aqi_category, aqi_value, analysis_data):
    """Get dynamic health recommendations based on AQI and specific analysis data"""
    
    # Base recommendations by category
    base_recommendations = {
        'Good': {
            'immediate_actions': [
                "🌟 Perfect air quality - no immediate actions needed",
                "🚶‍♂️ Enjoy outdoor activities freely",
                "🏃‍♀️ Great time for exercise and sports"
            ],
            'protective_measures': [
                "😷 No masks required",
                "🪟 Keep windows open for fresh air",
                "🌱 Consider outdoor plants and gardening"
            ],
            'activity_guidelines': [
                "🏃‍♂️ All outdoor activities recommended",
                "👶 Safe for children and elderly",
                "🏋️‍♀️ Perfect for intense workouts"
            ]
        },
        'Moderate': {
            'immediate_actions': [
                "⚠️ Monitor air quality throughout the day",
                "👥 Check on sensitive family members",
                "📱 Keep air quality app handy"
            ],
            'protective_measures': [
                "😷 Light masks for sensitive individuals",
                "🪟 Limit window opening during peak hours",
                "🌿 Use indoor air purifying plants"
            ],
            'activity_guidelines': [
                "🚶‍♂️ Reduce prolonged outdoor exercise",
                "👶 Monitor children during outdoor play",
                "🏃‍♀️ Consider indoor alternatives for workouts"
            ]
        },
        'Unhealthy for Sensitive Groups': {
            'immediate_actions': [
                "🚨 Sensitive groups should limit outdoor exposure",
                "💊 Have rescue medications readily available",
                "📞 Contact healthcare provider if symptoms occur"
            ],
            'protective_measures': [
                "😷 N95 masks recommended for sensitive groups",
                "🪟 Keep windows closed, use air conditioning",
                "🔧 Run air purifiers on high settings"
            ],
            'activity_guidelines': [
                "🏠 Move exercise indoors",
                "👶 Keep children and elderly indoors",
                "🚫 Avoid strenuous outdoor activities"
            ]
        },
        'Unhealthy': {
            'immediate_actions': [
                "🚨 Everyone should limit outdoor exposure",
                "💊 Have medications ready for respiratory conditions",
                "🏥 Seek medical attention for breathing difficulties"
            ],
            'protective_measures': [
                "😷 N95 or P100 masks when outdoors",
                "🪟 Seal windows and doors",
                "🔧 Use HEPA air purifiers continuously"
            ],
            'activity_guidelines': [
                "🏠 All activities should be indoors",
                "🚫 Cancel outdoor events and sports",
                "👥 Avoid crowded outdoor areas"
            ]
        },
        'Very Unhealthy': {
            'immediate_actions': [
                "🚨 EMERGENCY: Stay indoors immediately",
                "💊 Use prescribed inhalers/medications",
                "🏥 Seek immediate medical care for symptoms"
            ],
            'protective_measures': [
                "😷 P100 respirators required outdoors",
                "🔒 Completely seal living spaces",
                "🔧 Multiple HEPA purifiers running"
            ],
            'activity_guidelines': [
                "🏠 Mandatory indoor confinement",
                "🚫 Cancel all outdoor activities",
                "🚨 Emergency protocols for sensitive individuals"
            ]
        },
        'Hazardous': {
            'immediate_actions': [
                "🚨 EXTREME EMERGENCY: Shelter in place",
                "☎️ Call emergency services if experiencing symptoms",
                "💊 Use all prescribed emergency medications"
            ],
            'protective_measures': [
                "😷 Full-face respirators with P100 filters",
                "🔒 Create sealed safe room with air purification",
                "🚫 Absolutely no outdoor exposure"
            ],
            'activity_guidelines': [
                "🏠 Complete indoor isolation required",
                "🚨 Evacuate area if possible",
                "🏥 Emergency medical attention for any symptoms"
            ]
        }
    }
    
    # Get base recommendations
    recommendations = base_recommendations.get(aqi_category, base_recommendations['Hazardous'])
    
    # Add dynamic recommendations based on specific analysis
    dynamic_additions = []
    
    # Visibility-based recommendations
    if analysis_data.get('visibility', {}).get('visibility_score', 1) < 0.3:
        dynamic_additions.extend([
            "👁️ Very poor visibility detected - avoid driving",
            "🚗 Use headlights and hazard lights if must travel"
        ])
    
    # Haze-based recommendations
    if analysis_data.get('haze', {}).get('haze_density', 0) > 0.7:
        dynamic_additions.extend([
            "🌫️ Dense haze detected - respiratory protection critical",
            "💨 Avoid outdoor breathing exercises"
        ])
    
    # Particulate-based recommendations
    if analysis_data.get('particulates', {}).get('particulate_score', 0) > 0.6:
        dynamic_additions.extend([
            "🦠 High particulate matter - use N95+ masks",
            "🏠 Seal gaps around doors and windows"
        ])
    
    # Sky pollution recommendations
    if analysis_data.get('sky', {}).get('pollution_tint', 0) > 0.4:
        dynamic_additions.extend([
            "🎨 Visible pollution coloration - chemical irritants present",
            "👁️ Protect eyes with wraparound sunglasses"
        ])
    
    # Add dynamic recommendations to appropriate categories
    if dynamic_additions:
        recommendations['immediate_actions'].extend(dynamic_additions[:2])
        recommendations['protective_measures'].extend(dynamic_additions[2:4])
        recommendations['activity_guidelines'].extend(dynamic_additions[4:])
    
    return recommendations

def create_export_data(aqi_result, analysis_data, recommendations):
    """Create exportable data structure"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    export_data = {
        'analysis_timestamp': timestamp,
        'aqi_assessment': {
            'estimated_aqi': aqi_result['estimated_aqi'],
            'category': aqi_result['category'],
            'confidence': aqi_result['confidence']
        },
        'detailed_analysis': {
            'visibility': {
                'score': analysis_data['visibility']['visibility_score'],
                'interpretation': analysis_data['visibility']['interpretation']
            },
            'haze_level': {
                'density': analysis_data['haze']['haze_density'],
                'interpretation': analysis_data['haze']['interpretation']
            },
            'sky_analysis': {
                'pollution_tint': analysis_data['sky']['pollution_tint'],
                'rgb_values': analysis_data['sky']['sky_rgb'],
                'interpretation': analysis_data['sky']['interpretation']
            },
            'particulate_matter': {
                'score': analysis_data['particulates']['particulate_score'],
                'interpretation': analysis_data['particulates']['interpretation']
            }
        },
        'health_recommendations': recommendations
    }
    
    return export_data

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📸 Air Quality Image Analyzer</h1>
        <p>Upload or capture images to analyze air quality and get instant health recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = ImageAirQualityAnalyzer()
    
    # Sidebar for image input options
    st.sidebar.header("📤 Image Input")
    
    # Image input options
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["📁 Upload Image", "📷 Camera Capture", "🔗 URL Input"]
    )
    
    # Settings
    st.sidebar.header("⚙️ Settings")
    auto_analyze = st.sidebar.checkbox("🔄 Auto-analyze on upload", value=True)
    show_tips = st.sidebar.checkbox("💡 Show analysis tips", value=True)
    
    uploaded_file = None
    image = None
    
    # Handle different input methods
    if input_method == "📁 Upload Image":
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear outdoor image for air quality analysis"
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
    
    elif input_method == "📷 Camera Capture":
        camera_image = st.sidebar.camera_input("Take a picture of outdoor environment")
        if camera_image:
            image = Image.open(camera_image)
            uploaded_file = camera_image
    
    elif input_method == "🔗 URL Input":
        image_url = st.sidebar.text_input("Enter image URL:")
        if image_url:
            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
                uploaded_file = "url_image"
            except Exception as e:
                st.sidebar.error(f"Error loading image: {str(e)}")
    
    if show_tips:
        st.sidebar.header("💡 Tips for Best Results")
        st.sidebar.info("""
        📸 **Use outdoor images with visible sky**
        🌅 **Daytime photos work better**
        🏙️ **Include horizon/cityscape**
        📱 **Higher resolution = better analysis**
        """)
    
    if image is not None:
        # Main layout
        col1, col2 = st.columns([1.2, 0.8])
        
        with col1:
            st.markdown("### 📷 Image Preview")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("### 📊 Analysis Results")
            
            # Analysis status
            if auto_analyze:
                status_placeholder = st.empty()
                status_placeholder.markdown("""
                <div class="status-indicator status-analyzing">
                    🔄 Auto-analyzing image...
                </div>
                """, unsafe_allow_html=True)
                
                # Perform analysis
                progress_bar = st.progress(0)
                
                progress_bar.progress(25)
                visibility = analyzer.analyze_visibility(image)
                
                progress_bar.progress(50)
                haze = analyzer.analyze_haze_smog(image)
                
                progress_bar.progress(75)
                sky = analyzer.analyze_sky_pollution(image)
                particulates = analyzer.detect_particulates(image)
                
                progress_bar.progress(100)
                aqi_result = analyzer.estimate_aqi_from_visual(visibility, haze, sky, particulates)
                
                # Update status
                timestamp = datetime.now().strftime("%H:%M:%S")
                status_placeholder.markdown(f"""
                <div class="status-indicator status-complete">
                    ✅ Analysis completed at {timestamp}
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar.empty()
                
                # Display AQI result
                st.markdown(f"""
                <div class="aqi-display">
                    <h1>AQI: {aqi_result['estimated_aqi']}</h1>
                    <h3>{aqi_result['category']}</h3>
                    <p>Confidence: {aqi_result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed Analysis Section
        st.markdown("### 💎 Detailed Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            visibility_pct = int(visibility['visibility_score'] * 100)
            st.markdown(f"""
            <div class="detail-card">
                <h4>🔍 Atmospheric Visibility</h4>
                <h2>{visibility_pct}%</h2>
                <p>Higher percentage indicates clearer air</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sky_rgb = sky['sky_rgb']
            st.markdown(f"""
            <div class="detail-card">
                <h4>🎨 Sky Color Analysis</h4>
                <h2>RGB({sky_rgb[0]}, {sky_rgb[1]}, {sky_rgb[2]})</h2>
                <p>Pollution Tint: {sky['pollution_tint']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            haze_pct = int(haze['haze_density'] * 100)
            st.markdown(f"""
            <div class="detail-card">
                <h4>🌫️ Haze Level</h4>
                <h2>{haze_pct}%</h2>
                <p>Higher percentage indicates more haze/smog</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            particulate_pct = int(particulates['particulate_score'] * 100)
            st.markdown(f"""
            <div class="detail-card">
                <h4>🦠 Particulate Matter</h4>
                <h2>{particulate_pct}%</h2>
                <p>Visible particles and dust in air</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Health Recommendations Section
        st.markdown("### 🏥 Health Recommendations")
        
        analysis_data = {
            'visibility': visibility,
            'haze': haze,
            'sky': sky,
            'particulates': particulates
        }
        
        recommendations = get_dynamic_health_recommendations(
            aqi_result['category'], 
            aqi_result['estimated_aqi'], 
            analysis_data
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="immediate-action">
                <h4>⚡ Immediate Actions</h4>
            </div>
            """, unsafe_allow_html=True)
            for action in recommendations['immediate_actions']:
                st.write(f"• {action}")
        
        with col2:
            st.markdown("""
            <div class="protective-measure">
                <h4>🛡️ Protective Measures</h4>
            </div>
            """, unsafe_allow_html=True)
            for measure in recommendations['protective_measures']:
                st.write(f"• {measure}")
        
        with col3:
            st.markdown("""
            <div class="activity-guideline">
                <h4>🏃 Activity Guidelines</h4>
            </div>
            """, unsafe_allow_html=True)
            for guideline in recommendations['activity_guidelines']:
                st.write(f"• {guideline}")
        
        # Export Results Section
        st.markdown("### 📋 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        export_data = create_export_data(aqi_result, analysis_data, recommendations)
        
        with col1:
            if st.button("📄 Download JSON Report", key="json_export"):
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="💾 Save JSON File",
                    data=json_str,
                    file_name=f"air_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Download Summary", key="summary_export"):
                summary = f"""
Air Quality Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

AQI Assessment: {aqi_result['estimated_aqi']} ({aqi_result['category']})
Confidence: {aqi_result['confidence']:.1%}

Analysis Details:
- Visibility: {visibility['visibility_score']:.2f}
- Haze Level: {haze['haze_density']:.2f}
- Pollution Tint: {sky['pollution_tint']:.2f}
- Particulates: {particulates['particulate_score']:.2f}

Immediate Actions:
{chr(10).join(['- ' + action for action in recommendations['immediate_actions']])}
                """
                st.download_button(
                    label="💾 Save Summary",
                    data=summary,
                    file_name=f"air_quality_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("📈 Download CSV Data", key="csv_export"):
                csv_data = pd.DataFrame([{
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'AQI': aqi_result['estimated_aqi'],
                    'Category': aqi_result['category'],
                    'Confidence': aqi_result['confidence'],
                    'Visibility_Score': visibility['visibility_score'],
                    'Haze_Density': haze['haze_density'],
                    'Pollution_Tint': sky['pollution_tint'],
                    'Particulate_Score': particulates['particulate_score']
                }])
                csv_str = csv_data.to_csv(index=False)
                st.download_button(
                    label="💾 Save CSV File",
                    data=csv_str,
                    file_name=f"air_quality_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    else:
        # Welcome screen with instructions
        st.markdown("### 🌟 Welcome to Air Quality Image Analyzer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="health-recommendation">
                <h4>📸 Step 1: Capture/Upload</h4>
                <p>Take a picture of outdoor environment or upload an existing image</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="health-recommendation">
                <h4>🔍 Step 2: AI Analysis</h4>
                <p>Advanced computer vision analyzes air quality indicators</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="health-recommendation">
                <h4>🏥 Step 3: Get Recommendations</h4>
                <p>Receive personalized health and activity guidance</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📋 How It Works")
        st.write("""
        Our AI analyzes multiple visual indicators in your image:
        
        🔍 **Visibility Analysis** - Measures atmospheric clarity and contrast
        🌫️ **Haze Detection** - Identifies smog and particulate matter in the air  
        🎨 **Sky Color Analysis** - Detects pollution-related color changes
        🦠 **Particulate Detection** - Spots visible dust and particles
        
        Based on these factors, we estimate the Air Quality Index (AQI) and provide:
        ⚡ Immediate action recommendations
        🛡️ Protective measures to take
        🏃 Activity guidelines for your safety
        """)

if __name__ == "__main__":
    main()