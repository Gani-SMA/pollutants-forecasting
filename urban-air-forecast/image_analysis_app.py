import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageStat
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import base64
from io import BytesIO
import requests

# Page configuration
st.set_page_config(
    page_title="Air Quality Image Analysis",
    page_icon="📸",
    layout="wide"
)

# Custom CSS
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
    
    .analysis-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .aqi-good { background-color: #48bb78; }
    .aqi-moderate { background-color: #ed8936; }
    .aqi-unhealthy-sensitive { background-color: #f56565; }
    .aqi-unhealthy { background-color: #e53e3e; }
    .aqi-very-unhealthy { background-color: #9f7aea; }
    .aqi-hazardous { background-color: #742a2a; }
    
    .precaution-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

class ImageAirQualityAnalyzer:
    """Advanced image-based air quality analyzer"""
    
    def __init__(self):
        pass
    
    def analyze_visibility(self, image):
        """Analyze atmospheric visibility from image"""
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast (visibility indicator)
        contrast = np.std(gray) / 255.0
        
        # Edge detection for detail visibility
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Visibility score (0-1, higher = better visibility)
        visibility_score = min(1.0, (contrast * 0.7 + edge_density * 0.3) * 2)
        
        return {
            'visibility_score': visibility_score,
            'contrast': contrast,
            'edge_density': edge_density,
            'interpretation': self.interpret_visibility(visibility_score)
        }
    
    def analyze_haze_smog(self, image):
        """Analyze haze and smog levels"""
        img_array = np.array(image)
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Analyze saturation (haze reduces saturation)
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation) / 255.0
        
        # Analyze brightness uniformity
        value = hsv[:, :, 2]
        brightness_std = np.std(value) / 255.0
        
        # Calculate haze density
        haze_density = 1.0 - (avg_saturation * 0.6 + brightness_std * 0.4)
        haze_density = max(0, min(1, haze_density))
        
        return {
            'haze_density': haze_density,
            'avg_saturation': avg_saturation,
            'brightness_variation': brightness_std,
            'interpretation': self.interpret_haze(haze_density)
        }
    
    def analyze_sky_pollution(self, image):
        """Analyze sky color for pollution indicators"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Focus on upper portion (sky region)
        sky_region = img_array[:height//3, :, :]
        
        # Calculate average RGB
        avg_r = np.mean(sky_region[:, :, 0])
        avg_g = np.mean(sky_region[:, :, 1])
        avg_b = np.mean(sky_region[:, :, 2])
        
        # Calculate pollution indicators
        # Brown/yellow tint indicates pollution
        pollution_tint = (avg_r + avg_g - avg_b) / (avg_r + avg_g + avg_b + 1)
        
        # Gray level (desaturation indicates pollution)
        gray_level = 1 - (max(avg_r, avg_g, avg_b) - min(avg_r, avg_g, avg_b)) / 255.0
        
        return {
            'sky_rgb': [avg_r, avg_g, avg_b],
            'pollution_tint': max(0, pollution_tint),
            'gray_level': gray_level,
            'interpretation': self.interpret_sky_color(pollution_tint, gray_level)
        }
    
    def detect_particulates(self, image):
        """Detect visible particulates and dust"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply noise detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_variance = np.var(laplacian)
        
        # Texture analysis for particulates
        kernel = np.ones((5, 5), np.float32) / 25
        mean_filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        texture = np.mean(np.abs(gray.astype(np.float32) - mean_filtered))
        
        # Normalize scores
        noise_score = min(1.0, noise_variance / 10000)
        texture_score = min(1.0, texture / 50)
        
        particulate_score = (noise_score + texture_score) / 2
        
        return {
            'particulate_score': particulate_score,
            'noise_variance': noise_variance,
            'texture_measure': texture,
            'interpretation': self.interpret_particulates(particulate_score)
        }
    
    def estimate_aqi_from_visual(self, visibility, haze, sky, particulates):
        """Estimate AQI from visual indicators"""
        # Weighted combination of indicators
        pollution_index = (
            (1 - visibility['visibility_score']) * 0.25 +
            haze['haze_density'] * 0.30 +
            sky['pollution_tint'] * 0.20 +
            particulates['particulate_score'] * 0.25
        )
        
        # Map to AQI scale (0-500)
        estimated_aqi = pollution_index * 350  # Scale to realistic AQI range
        estimated_aqi = max(10, min(500, estimated_aqi))
        
        # Determine category
        if estimated_aqi <= 50:
            category, color = "Good", "green"
        elif estimated_aqi <= 100:
            category, color = "Moderate", "yellow"
        elif estimated_aqi <= 150:
            category, color = "Unhealthy for Sensitive Groups", "orange"
        elif estimated_aqi <= 200:
            category, color = "Unhealthy", "red"
        elif estimated_aqi <= 300:
            category, color = "Very Unhealthy", "purple"
        else:
            category, color = "Hazardous", "maroon"
        
        return {
            'estimated_aqi': estimated_aqi,
            'category': category,
            'color': color,
            'pollution_index': pollution_index,
            'confidence': self.calculate_confidence(visibility, haze, sky, particulates)
        }
    
    def calculate_confidence(self, visibility, haze, sky, particulates):
        """Calculate confidence in the analysis"""
        # Higher confidence when indicators agree
        indicators = [
            1 - visibility['visibility_score'],
            haze['haze_density'],
            sky['pollution_tint'],
            particulates['particulate_score']
        ]
        
        # Calculate agreement (lower std = higher confidence)
        agreement = 1 - (np.std(indicators) / np.mean(indicators) if np.mean(indicators) > 0 else 0)
        confidence = max(0.3, min(0.95, agreement))
        
        return confidence
    
    def interpret_visibility(self, score):
        if score > 0.8: return "Excellent visibility - very clear air"
        elif score > 0.6: return "Good visibility - minor haze"
        elif score > 0.4: return "Moderate visibility - noticeable haze"
        elif score > 0.2: return "Poor visibility - significant pollution"
        else: return "Very poor visibility - heavy pollution"
    
    def interpret_haze(self, density):
        if density < 0.2: return "Clear air - no significant haze"
        elif density < 0.4: return "Light haze - minor pollution"
        elif density < 0.6: return "Moderate haze - noticeable pollution"
        elif density < 0.8: return "Heavy haze - significant pollution"
        else: return "Very heavy haze - severe pollution"
    
    def interpret_sky_color(self, tint, gray):
        if tint < 0.1 and gray < 0.3: return "Natural sky color - clean air"
        elif tint < 0.3: return "Slight discoloration - minor pollution"
        elif tint < 0.5: return "Noticeable discoloration - moderate pollution"
        else: return "Significant discoloration - heavy pollution"
    
    def interpret_particulates(self, score):
        if score < 0.2: return "Low particulate levels"
        elif score < 0.4: return "Moderate particulate levels"
        elif score < 0.6: return "High particulate levels"
        else: return "Very high particulate levels"
    
    def generate_health_recommendations(self, aqi_estimate):
        """Generate health recommendations based on estimated AQI"""
        aqi = aqi_estimate['estimated_aqi']
        category = aqi_estimate['category']
        
        recommendations = {
            'immediate_actions': [],
            'protective_measures': [],
            'activity_modifications': [],
            'vulnerable_groups': []
        }
        
        if aqi <= 50:  # Good
            recommendations['immediate_actions'] = [
                "No immediate actions needed",
                "Enjoy outdoor activities normally"
            ]
            recommendations['activity_modifications'] = [
                "All outdoor activities are safe",
                "Good time for exercise and sports"
            ]
        
        elif aqi <= 100:  # Moderate
            recommendations['immediate_actions'] = [
                "Monitor air quality if you're sensitive to pollution",
                "Consider reducing time outdoors if experiencing symptoms"
            ]
            recommendations['vulnerable_groups'] = [
                "People with respiratory conditions should be cautious",
                "Children and elderly should limit prolonged outdoor exposure"
            ]
        
        elif aqi <= 150:  # Unhealthy for Sensitive Groups
            recommendations['immediate_actions'] = [
                "Sensitive individuals should limit outdoor activities",
                "Close windows and use air purifiers indoors"
            ]
            recommendations['protective_measures'] = [
                "Consider wearing N95 masks outdoors",
                "Use air purifiers in indoor spaces"
            ]
            recommendations['vulnerable_groups'] = [
                "Children, elderly, and people with heart/lung disease should stay indoors",
                "Avoid outdoor exercise and strenuous activities"
            ]
        
        elif aqi <= 200:  # Unhealthy
            recommendations['immediate_actions'] = [
                "Everyone should limit outdoor activities",
                "Stay indoors with windows closed"
            ]
            recommendations['protective_measures'] = [
                "Wear N95 or P100 masks when outdoors",
                "Use high-efficiency air purifiers",
                "Avoid opening windows"
            ]
            recommendations['activity_modifications'] = [
                "Cancel outdoor events and activities",
                "Move exercise indoors"
            ]
        
        elif aqi <= 300:  # Very Unhealthy
            recommendations['immediate_actions'] = [
                "Avoid outdoor activities entirely",
                "Stay indoors and seal windows/doors"
            ]
            recommendations['protective_measures'] = [
                "Use highest grade masks (P100) if must go outside",
                "Run multiple air purifiers indoors",
                "Consider temporary relocation"
            ]
            recommendations['vulnerable_groups'] = [
                "Seek medical attention if experiencing symptoms",
                "Consider evacuation for sensitive individuals"
            ]
        
        else:  # Hazardous
            recommendations['immediate_actions'] = [
                "Emergency conditions - stay indoors",
                "Avoid all outdoor exposure"
            ]
            recommendations['protective_measures'] = [
                "Use professional-grade respiratory protection",
                "Seal all openings, use industrial air purifiers",
                "Consider evacuation if possible"
            ]
            recommendations['vulnerable_groups'] = [
                "Seek immediate medical attention for any symptoms",
                "Emergency evacuation recommended for sensitive groups"
            ]
        
        return recommendations

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📸 Air Quality Image Analysis</h1>
        <p>Analyze air pollution from photos using computer vision</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ImageAirQualityAnalyzer()
    
    # Sidebar instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.write("""
        **For best results:**
        - Take photos outdoors during daylight
        - Include sky and distant objects
        - Avoid direct sunlight or shadows
        - Ensure image is clear and focused
        """)
        
        st.header("📍 Location Info")
        location = st.text_input("Location (optional):", placeholder="e.g., Delhi, India")
        
        st.header("⚙️ Analysis Options")
        analyze_all = st.checkbox("Analyze All Indicators", value=True)
        
        if not analyze_all:
            analyze_visibility = st.checkbox("🔍 Visibility Analysis", value=True)
            analyze_haze = st.checkbox("🌫️ Haze/Smog Analysis", value=True)
            analyze_sky = st.checkbox("🌤️ Sky Color Analysis", value=True)
            analyze_particles = st.checkbox("💨 Particulate Detection", value=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📸 Image Upload", "📊 Analysis Results", "🏥 Health Recommendations"])
    
    with tab1:
        st.header("📸 Upload Image for Analysis")
        
        # Image input methods
        input_method = st.radio(
            "Choose input method:",
            ["📁 Upload from Device", "📷 Camera Capture", "🌐 Image URL", "🎯 Sample Images"]
        )
        
        image = None
        
        if input_method == "📁 Upload from Device":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload a clear outdoor image showing the sky and surroundings"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        
        elif input_method == "📷 Camera Capture":
            st.write("📱 Use your device camera to capture the current air quality conditions")
            camera_image = st.camera_input("Take a picture of the outdoor environment")
            
            if camera_image is not None:
                image = Image.open(camera_image)
        
        elif input_method == "🌐 Image URL":
            image_url = st.text_input("Enter image URL:")
            
            if image_url:
                try:
                    response = requests.get(image_url, timeout=10)
                    image = Image.open(BytesIO(response.content))
                    st.success("✅ Image loaded from URL")
                except Exception as e:
                    st.error(f"❌ Error loading image: {str(e)}")
        
        elif input_method == "🎯 Sample Images":
            st.write("Select a sample image to test the analysis:")
            
            sample_options = {
                "Clear Day": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
                "Hazy Conditions": "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=800",
                "Smoggy City": "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=800"
            }
            
            selected_sample = st.selectbox("Choose sample:", list(sample_options.keys()))
            
            if st.button("Load Sample Image"):
                try:
                    response = requests.get(sample_options[selected_sample], timeout=10)
                    image = Image.open(BytesIO(response.content))
                    st.success(f"✅ Loaded sample: {selected_sample}")
                except Exception as e:
                    st.error(f"❌ Error loading sample: {str(e)}")
        
        # Display image and basic info
        if image is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(image, caption="Image for Analysis", use_column_width=True)
            
            with col2:
                st.write("**Image Information:**")
                st.write(f"📏 Size: {image.size[0]} × {image.size[1]} pixels")
                st.write(f"🎨 Mode: {image.mode}")
                st.write(f"📊 Format: {image.format if hasattr(image, 'format') else 'Unknown'}")
                
                # Image quality check
                if image.size[0] < 300 or image.size[1] < 300:
                    st.warning("⚠️ Image resolution is low. Results may be less accurate.")
                
                if image.mode != 'RGB':
                    st.info("🔄 Converting image to RGB format")
                    image = image.convert('RGB')
            
            # Analysis button
            if st.button("🔬 Analyze Air Quality", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing image for air quality indicators..."):
                    # Perform analysis
                    visibility_results = st.session_state.analyzer.analyze_visibility(image)
                    haze_results = st.session_state.analyzer.analyze_haze_smog(image)
                    sky_results = st.session_state.analyzer.analyze_sky_pollution(image)
                    particulate_results = st.session_state.analyzer.detect_particulates(image)
                    
                    # Estimate AQI
                    aqi_estimate = st.session_state.analyzer.estimate_aqi_from_visual(
                        visibility_results, haze_results, sky_results, particulate_results
                    )
                    
                    # Store results
                    st.session_state.analysis_results = {
                        'timestamp': datetime.now().isoformat(),
                        'location': location,
                        'visibility': visibility_results,
                        'haze': haze_results,
                        'sky': sky_results,
                        'particulates': particulate_results,
                        'aqi_estimate': aqi_estimate
                    }
                    
                    st.success("✅ Analysis completed! Check the 'Analysis Results' tab.")
    
    with tab2:
        if hasattr(st.session_state, 'analysis_results'):
            results = st.session_state.analysis_results
            
            st.header("📊 Analysis Results")
            
            # Main AQI estimate
            aqi_est = results['aqi_estimate']
            
            st.markdown(f"""
            <div class="aqi-{aqi_est['color']}" style="color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
                <h1>Estimated AQI: {aqi_est['estimated_aqi']:.0f}</h1>
                <h2>{aqi_est['category']}</h2>
                <p>Confidence: {aqi_est['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed metrics
            st.subheader("🔍 Detailed Analysis")
            
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                vis_score = results['visibility']['visibility_score']
                st.metric(
                    "👁️ Visibility",
                    f"{vis_score:.2f}",
                    delta=results['visibility']['interpretation']
                )
            
            with metric_cols[1]:
                haze_density = results['haze']['haze_density']
                st.metric(
                    "🌫️ Haze Density",
                    f"{haze_density:.2f}",
                    delta=results['haze']['interpretation']
                )
            
            with metric_cols[2]:
                pollution_tint = results['sky']['pollution_tint']
                st.metric(
                    "🌤️ Sky Pollution",
                    f"{pollution_tint:.2f}",
                    delta=results['sky']['interpretation']
                )
            
            with metric_cols[3]:
                particle_score = results['particulates']['particulate_score']
                st.metric(
                    "💨 Particulates",
                    f"{particle_score:.2f}",
                    delta=results['particulates']['interpretation']
                )
            
            # Radar chart visualization
            st.subheader("📈 Pollution Indicators Radar")
            
            categories = ['Visibility Loss', 'Haze Density', 'Sky Pollution', 'Particulates']
            values = [
                1 - results['visibility']['visibility_score'],
                results['haze']['haze_density'],
                results['sky']['pollution_tint'],
                results['particulates']['particulate_score']
            ]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Pollution Level',
                line_color='red',
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        ticktext=['Clean', 'Light', 'Moderate', 'High', 'Severe', 'Extreme']
                    )
                ),
                showlegend=True,
                title="Visual Pollution Assessment",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sky color visualization
            st.subheader("🎨 Sky Color Analysis")
            
            sky_rgb = results['sky']['sky_rgb']
            rgb_normalized = [int(c) for c in sky_rgb]
            
            st.markdown(f"""
            <div style="background-color: rgb({rgb_normalized[0]}, {rgb_normalized[1]}, {rgb_normalized[2]}); 
                        height: 100px; border-radius: 10px; margin: 10px 0;
                        display: flex; align-items: center; justify-content: center; 
                        color: {'white' if sum(rgb_normalized) < 400 else 'black'}; font-weight: bold;">
                Average Sky Color<br>
                RGB({rgb_normalized[0]}, {rgb_normalized[1]}, {rgb_normalized[2]})
            </div>
            """, unsafe_allow_html=True)
            
            # Technical details
            with st.expander("🔧 Technical Analysis Details"):
                st.json(results)
        
        else:
            st.info("📸 Upload and analyze an image first to see results here.")
    
    with tab3:
        if hasattr(st.session_state, 'analysis_results'):
            results = st.session_state.analysis_results
            aqi_est = results['aqi_estimate']
            
            st.header("🏥 Health Recommendations")
            
            # Generate recommendations
            recommendations = st.session_state.analyzer.generate_health_recommendations(aqi_est)
            
            # Display AQI status
            st.markdown(f"""
            <div class="analysis-card">
                <h3>🎯 Current Air Quality Assessment</h3>
                <p><strong>Estimated AQI:</strong> {aqi_est['estimated_aqi']:.0f} ({aqi_est['category']})</p>
                <p><strong>Analysis Confidence:</strong> {aqi_est['confidence']:.1%}</p>
                <p><strong>Location:</strong> {results.get('location', 'Not specified')}</p>
                <p><strong>Analysis Time:</strong> {results['timestamp']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations by category
            rec_tabs = st.tabs(["🚨 Immediate Actions", "🛡️ Protective Measures", "🏃 Activity Changes", "⚠️ Vulnerable Groups"])
            
            with rec_tabs[0]:
                st.subheader("🚨 Immediate Actions")
                for action in recommendations['immediate_actions']:
                    st.write(f"• {action}")
            
            with rec_tabs[1]:
                st.subheader("🛡️ Protective Measures")
                for measure in recommendations['protective_measures']:
                    st.write(f"• {measure}")
            
            with rec_tabs[2]:
                st.subheader("🏃 Activity Modifications")
                for modification in recommendations['activity_modifications']:
                    st.write(f"• {modification}")
            
            with rec_tabs[3]:
                st.subheader("⚠️ Special Precautions for Vulnerable Groups")
                for precaution in recommendations['vulnerable_groups']:
                    st.write(f"• {precaution}")
            
            # Emergency contact info
            if aqi_est['estimated_aqi'] > 200:
                st.error("""
                🚨 **EMERGENCY LEVEL AIR POLLUTION DETECTED**
                
                - Contact local emergency services if experiencing breathing difficulties
                - Consider immediate indoor shelter or evacuation
                - Monitor official air quality alerts and warnings
                """)
            
            # Export options
            st.subheader("📤 Export Analysis")
            
            export_cols = st.columns(2)
            
            with export_cols[0]:
                # JSON export
                json_data = {
                    'analysis_results': results,
                    'health_recommendations': recommendations
                }
                
                json_str = json.dumps(json_data, indent=2, default=str)
                st.download_button(
                    label="📄 Download Full Report (JSON)",
                    data=json_str,
                    file_name=f"air_quality_image_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with export_cols[1]:
                # Summary report
                summary_report = f"""
# Air Quality Image Analysis Report

**Analysis Date:** {results['timestamp']}
**Location:** {results.get('location', 'Not specified')}

## Results Summary
- **Estimated AQI:** {aqi_est['estimated_aqi']:.0f}
- **Air Quality Category:** {aqi_est['category']}
- **Analysis Confidence:** {aqi_est['confidence']:.1%}

## Visual Indicators
- **Visibility Score:** {results['visibility']['visibility_score']:.2f}
- **Haze Density:** {results['haze']['haze_density']:.2f}
- **Sky Pollution Tint:** {results['sky']['pollution_tint']:.2f}
- **Particulate Score:** {results['particulates']['particulate_score']:.2f}

## Health Recommendations
### Immediate Actions
{chr(10).join([f"- {action}" for action in recommendations['immediate_actions']])}

### Protective Measures
{chr(10).join([f"- {measure}" for measure in recommendations['protective_measures']])}

### Activity Modifications
{chr(10).join([f"- {mod}" for mod in recommendations['activity_modifications']])}

### Vulnerable Groups
{chr(10).join([f"- {precaution}" for precaution in recommendations['vulnerable_groups']])}

---
*Generated by Air Quality Image Analysis System*
                """
                
                st.download_button(
                    label="📋 Download Summary Report",
                    data=summary_report,
                    file_name=f"air_quality_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        else:
            st.info("📸 Complete image analysis first to see health recommendations.")

if __name__ == "__main__":
    main()