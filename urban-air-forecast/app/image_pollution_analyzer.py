import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageStat
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import base64
from io import BytesIO
import logging
from pathlib import Path

class ImagePollutionAnalyzer:
    """Analyze air pollution from images using computer vision techniques"""
    
    def __init__(self):
        self.logger = self.setup_logging()
        self.pollution_thresholds = {
            'visibility': {
                'good': 0.8,
                'moderate': 0.6,
                'unhealthy_sensitive': 0.4,
                'unhealthy': 0.2,
                'very_unhealthy': 0.1
            },
            'haze_density': {
                'good': 0.1,
                'moderate': 0.3,
                'unhealthy_sensitive': 0.5,
                'unhealthy': 0.7,
                'very_unhealthy': 0.9
            }
        }
        
    def setup_logging(self):
        """Setup logging for image analysis"""
        log_dir = Path("urban-air-forecast/logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger('image_analyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(log_dir / "image_analysis.log")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_visibility(self, image):
        """Analyze visibility from image using contrast and edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Calculate contrast using standard deviation
        contrast = np.std(gray) / 255.0
        
        # Edge detection to measure detail visibility
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Combine metrics for visibility score
        visibility_score = (contrast * 0.6 + edge_density * 0.4)
        
        return {
            'visibility_score': float(visibility_score),
            'contrast': float(contrast),
            'edge_density': float(edge_density)
        }
    
    def analyze_haze_density(self, image):
        """Analyze haze/smog density using color analysis"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
        
        # Analyze saturation (haze reduces color saturation)
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation) / 255.0
        
        # Analyze value (brightness) distribution
        value = hsv[:, :, 2]
        brightness_std = np.std(value) / 255.0
        
        # Calculate haze density (inverse of saturation and brightness variation)
        haze_density = 1.0 - (avg_saturation * 0.7 + brightness_std * 0.3)
        
        return {
            'haze_density': float(haze_density),
            'avg_saturation': float(avg_saturation),
            'brightness_variation': float(brightness_std)
        }
    
    def analyze_sky_color(self, image):
        """Analyze sky color to detect pollution"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Focus on upper portion of image (likely sky)
        height = img_array.shape[0]
        sky_region = img_array[:height//3, :, :]
        
        # Calculate average RGB values
        avg_r = np.mean(sky_region[:, :, 0])
        avg_g = np.mean(sky_region[:, :, 1])
        avg_b = np.mean(sky_region[:, :, 2])
        
        # Calculate color ratios
        total = avg_r + avg_g + avg_b
        if total > 0:
            r_ratio = avg_r / total
            g_ratio = avg_g / total
            b_ratio = avg_b / total
        else:
            r_ratio = g_ratio = b_ratio = 0.33
        
        # Detect brownish/yellowish tint (pollution indicator)
        pollution_tint = (r_ratio + g_ratio - b_ratio) / 2
        
        return {
            'sky_color_rgb': [float(avg_r), float(avg_g), float(avg_b)],
            'color_ratios': [float(r_ratio), float(g_ratio), float(b_ratio)],
            'pollution_tint': float(pollution_tint)
        }
    
    def detect_particulates(self, image):
        """Detect visible particulates using texture analysis"""
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate local standard deviation (texture measure)
        kernel = np.ones((9, 9), np.float32) / 81
        mean = cv2.filter2D(blurred.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((blurred.astype(np.float32))**2, -1, kernel)
        texture = np.sqrt(sqr_mean - mean**2)
        
        # Calculate average texture (higher = more particulates)
        avg_texture = np.mean(texture) / 255.0
        
        # Detect high-frequency noise (dust/particulates)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = np.var(laplacian) / (255.0**2)
        
        return {
            'texture_measure': float(avg_texture),
            'noise_level': float(noise_level),
            'particulate_score': float((avg_texture + noise_level) / 2)
        }
    
    def estimate_aqi_from_image(self, visibility_score, haze_density, pollution_tint, particulate_score):
        """Estimate AQI based on visual indicators"""
        # Weighted combination of visual indicators
        visual_pollution_index = (
            (1 - visibility_score) * 0.3 +
            haze_density * 0.3 +
            pollution_tint * 0.2 +
            particulate_score * 0.2
        )
        
        # Map to AQI scale (0-500)
        estimated_aqi = min(500, max(0, visual_pollution_index * 400))
        
        # Determine AQI category
        if estimated_aqi <= 50:
            category = "Good"
            color = "green"
        elif estimated_aqi <= 100:
            category = "Moderate"
            color = "yellow"
        elif estimated_aqi <= 150:
            category = "Unhealthy for Sensitive Groups"
            color = "orange"
        elif estimated_aqi <= 200:
            category = "Unhealthy"
            color = "red"
        elif estimated_aqi <= 300:
            category = "Very Unhealthy"
            color = "purple"
        else:
            category = "Hazardous"
            color = "maroon"
        
        return {
            'estimated_aqi': float(estimated_aqi),
            'aqi_category': category,
            'aqi_color': color,
            'visual_pollution_index': float(visual_pollution_index)
        }
    
    def generate_precautions(self, aqi_estimate, analysis_results):
        """Generate health precautions based on image analysis"""
        aqi_value = aqi_estimate['estimated_aqi']
        category = aqi_estimate['aqi_category']
        
        precautions = {
            'general': [],
            'sensitive_groups': [],
            'outdoor_activities': [],
            'protective_measures': []
        }
        
        if aqi_value <= 50:  # Good
            precautions['general'] = [
                "Air quality is satisfactory for most people",
                "Enjoy outdoor activities normally"
            ]
            precautions['outdoor_activities'] = [
                "All outdoor activities are safe",
                "Good time for exercise and sports"
            ]
        
        elif aqi_value <= 100:  # Moderate
            precautions['general'] = [
                "Air quality is acceptable for most people",
                "Unusually sensitive people may experience minor symptoms"
            ]
            precautions['sensitive_groups'] = [
                "People with respiratory conditions should monitor symptoms",
                "Consider reducing prolonged outdoor exertion"
            ]
            precautions['outdoor_activities'] = [
                "Most outdoor activities are safe",
                "Sensitive individuals may want to limit prolonged outdoor activities"
            ]
        
        elif aqi_value <= 150:  # Unhealthy for Sensitive Groups
            precautions['general'] = [
                "Sensitive groups may experience health effects",
                "General public is less likely to be affected"
            ]
            precautions['sensitive_groups'] = [
                "Children, elderly, and people with heart/lung disease should limit outdoor activities",
                "Watch for symptoms like coughing or shortness of breath"
            ]
            precautions['outdoor_activities'] = [
                "Reduce prolonged or heavy outdoor exertion",
                "Take frequent breaks during outdoor activities"
            ]
            precautions['protective_measures'] = [
                "Consider wearing N95 masks outdoors",
                "Keep windows closed, use air purifiers indoors"
            ]
        
        elif aqi_value <= 200:  # Unhealthy
            precautions['general'] = [
                "Everyone may begin to experience health effects",
                "Sensitive groups may experience more serious effects"
            ]
            precautions['sensitive_groups'] = [
                "Avoid outdoor activities",
                "Stay indoors with windows and doors closed"
            ]
            precautions['outdoor_activities'] = [
                "Avoid prolonged or heavy outdoor exertion",
                "Consider moving activities indoors"
            ]
            precautions['protective_measures'] = [
                "Wear N95 or P100 masks when outdoors",
                "Use air purifiers indoors",
                "Avoid outdoor exercise"
            ]
        
        elif aqi_value <= 300:  # Very Unhealthy
            precautions['general'] = [
                "Health alert: everyone may experience serious health effects",
                "Avoid outdoor activities"
            ]
            precautions['sensitive_groups'] = [
                "Stay indoors and avoid physical activities",
                "Seek medical attention if experiencing symptoms"
            ]
            precautions['outdoor_activities'] = [
                "Avoid all outdoor activities",
                "Cancel outdoor events and sports"
            ]
            precautions['protective_measures'] = [
                "Wear high-quality masks (N95/P100) if must go outside",
                "Seal windows and doors, use air purifiers",
                "Consider relocating temporarily if possible"
            ]
        
        else:  # Hazardous
            precautions['general'] = [
                "Emergency conditions: everyone is at risk of serious health effects",
                "Stay indoors and avoid all outdoor activities"
            ]
            precautions['sensitive_groups'] = [
                "Remain indoors and avoid physical activities",
                "Seek immediate medical attention for any symptoms"
            ]
            precautions['outdoor_activities'] = [
                "All outdoor activities should be cancelled",
                "Emergency outdoor exposure only"
            ]
            precautions['protective_measures'] = [
                "Use highest grade respiratory protection (P100)",
                "Seal all openings, use multiple air purifiers",
                "Consider evacuation if conditions persist"
            ]
        
        # Add specific precautions based on analysis
        if analysis_results['haze_density'] > 0.7:
            precautions['protective_measures'].append("High haze detected - visibility is severely reduced")
        
        if analysis_results['particulate_score'] > 0.6:
            precautions['protective_measures'].append("High particulate levels detected - use respiratory protection")
        
        return precautions
    
    def analyze_image(self, image):
        """Complete image analysis pipeline"""
        try:
            self.logger.info("Starting image analysis")
            
            # Perform all analyses
            visibility_results = self.analyze_visibility(image)
            haze_results = self.analyze_haze_density(image)
            sky_results = self.analyze_sky_color(image)
            particulate_results = self.detect_particulates(image)
            
            # Estimate AQI
            aqi_estimate = self.estimate_aqi_from_image(
                visibility_results['visibility_score'],
                haze_results['haze_density'],
                sky_results['pollution_tint'],
                particulate_results['particulate_score']
            )
            
            # Combine all results
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'visibility': visibility_results,
                'haze': haze_results,
                'sky_color': sky_results,
                'particulates': particulate_results,
                'aqi_estimate': aqi_estimate
            }
            
            # Generate precautions
            precautions = self.generate_precautions(aqi_estimate, {
                'haze_density': haze_results['haze_density'],
                'particulate_score': particulate_results['particulate_score']
            })
            
            analysis_results['precautions'] = precautions
            
            self.logger.info(f"Image analysis completed - Estimated AQI: {aqi_estimate['estimated_aqi']:.1f}")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in image analysis: {str(e)}")
            return None
    
    def create_analysis_visualization(self, analysis_results):
        """Create visualizations for the analysis results"""
        if not analysis_results:
            return None
        
        # Create radar chart for pollution indicators
        categories = ['Visibility', 'Haze Density', 'Pollution Tint', 'Particulates']
        values = [
            1 - analysis_results['visibility']['visibility_score'],  # Invert for pollution scale
            analysis_results['haze']['haze_density'],
            analysis_results['sky_color']['pollution_tint'],
            analysis_results['particulates']['particulate_score']
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Pollution Indicators',
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Visual Pollution Indicators"
        )
        
        return fig

def create_image_analysis_ui():
    """Create Streamlit UI for image-based pollution analysis"""
    st.header("📸 Image-Based Air Quality Analysis")
    st.write("Upload or capture an image to analyze air quality based on visual indicators")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ImagePollutionAnalyzer()
    
    # Image input options
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload Image", "📷 Camera Capture", "🌐 URL Input"]
    )
    
    image = None
    
    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a clear outdoor image showing the sky and surroundings"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    
    elif input_method == "📷 Camera Capture":
        camera_image = st.camera_input("Take a picture of the outdoor environment")
        
        if camera_image is not None:
            image = Image.open(camera_image)
    
    elif input_method == "🌐 URL Input":
        image_url = st.text_input("Enter image URL:")
        
        if image_url:
            try:
                import requests
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                st.error(f"Error loading image from URL: {str(e)}")
    
    if image is not None:
        # Display the image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Input Image")
            st.image(image, caption="Image for Analysis", use_column_width=True)
            
            # Image info
            st.write(f"**Image Size:** {image.size[0]} × {image.size[1]} pixels")
            st.write(f"**Image Mode:** {image.mode}")
        
        with col2:
            st.subheader("⚙️ Analysis Settings")
            
            # Analysis options
            analyze_visibility = st.checkbox("🔍 Analyze Visibility", value=True)
            analyze_haze = st.checkbox("🌫️ Analyze Haze/Smog", value=True)
            analyze_sky = st.checkbox("🌤️ Analyze Sky Color", value=True)
            analyze_particles = st.checkbox("💨 Detect Particulates", value=True)
            
            # Location context (optional)
            location = st.text_input("📍 Location (optional):", placeholder="e.g., Delhi, India")
            
            # Analysis button
            if st.button("🔬 Analyze Image", type="primary"):
                with st.spinner("Analyzing image for air quality indicators..."):
                    # Perform analysis
                    analysis_results = st.session_state.analyzer.analyze_image(image)
                    
                    if analysis_results:
                        st.session_state.analysis_results = analysis_results
                        st.session_state.analysis_location = location
                        st.success("✅ Analysis completed!")
                    else:
                        st.error("❌ Analysis failed. Please try with a different image.")
    
    # Display results if available
    if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
        display_analysis_results(st.session_state.analysis_results, st.session_state.get('analysis_location', ''))

def display_analysis_results(results, location):
    """Display comprehensive analysis results"""
    st.header("📊 Analysis Results")
    
    # AQI Estimate
    aqi_estimate = results['aqi_estimate']
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Estimated AQI",
            f"{aqi_estimate['estimated_aqi']:.0f}",
            delta=aqi_estimate['aqi_category']
        )
    
    with col2:
        st.metric(
            "👁️ Visibility Score",
            f"{results['visibility']['visibility_score']:.2f}",
            delta="Good" if results['visibility']['visibility_score'] > 0.7 else "Poor"
        )
    
    with col3:
        st.metric(
            "🌫️ Haze Density",
            f"{results['haze']['haze_density']:.2f}",
            delta="High" if results['haze']['haze_density'] > 0.5 else "Low"
        )
    
    with col4:
        st.metric(
            "💨 Particulate Level",
            f"{results['particulates']['particulate_score']:.2f}",
            delta="High" if results['particulates']['particulate_score'] > 0.5 else "Low"
        )
    
    # AQI Category Display
    category_color = aqi_estimate['aqi_color']
    st.markdown(f"""
    <div style="background-color: {category_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
        <h2>Air Quality: {aqi_estimate['aqi_category']}</h2>
        <h3>AQI: {aqi_estimate['estimated_aqi']:.0f}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed Analysis
    st.subheader("🔍 Detailed Analysis")
    
    analysis_tabs = st.tabs(["👁️ Visibility", "🌫️ Haze Analysis", "🌤️ Sky Color", "💨 Particulates"])
    
    with analysis_tabs[0]:
        vis_data = results['visibility']
        st.write("**Visibility Analysis:**")
        st.write(f"- Overall Visibility Score: {vis_data['visibility_score']:.3f}")
        st.write(f"- Image Contrast: {vis_data['contrast']:.3f}")
        st.write(f"- Edge Density: {vis_data['edge_density']:.3f}")
        
        # Visibility interpretation
        if vis_data['visibility_score'] > 0.7:
            st.success("✅ Good visibility - clear atmospheric conditions")
        elif vis_data['visibility_score'] > 0.4:
            st.warning("⚠️ Moderate visibility - some atmospheric haze present")
        else:
            st.error("❌ Poor visibility - significant atmospheric pollution")
    
    with analysis_tabs[1]:
        haze_data = results['haze']
        st.write("**Haze/Smog Analysis:**")
        st.write(f"- Haze Density: {haze_data['haze_density']:.3f}")
        st.write(f"- Average Saturation: {haze_data['avg_saturation']:.3f}")
        st.write(f"- Brightness Variation: {haze_data['brightness_variation']:.3f}")
        
        # Haze interpretation
        if haze_data['haze_density'] < 0.3:
            st.success("✅ Low haze levels - clear air")
        elif haze_data['haze_density'] < 0.6:
            st.warning("⚠️ Moderate haze - some air pollution present")
        else:
            st.error("❌ High haze levels - significant air pollution")
    
    with analysis_tabs[2]:
        sky_data = results['sky_color']
        st.write("**Sky Color Analysis:**")
        st.write(f"- Average Sky Color (RGB): {sky_data['sky_color_rgb']}")
        st.write(f"- Color Ratios (R:G:B): {[f'{r:.2f}' for r in sky_data['color_ratios']]}")
        st.write(f"- Pollution Tint Score: {sky_data['pollution_tint']:.3f}")
        
        # Color visualization
        rgb_color = [int(c) for c in sky_data['sky_color_rgb']]
        st.markdown(f"""
        <div style="background-color: rgb({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}); 
                    height: 50px; border-radius: 5px; margin: 10px 0;
                    display: flex; align-items: center; justify-content: center; color: white;">
            Average Sky Color
        </div>
        """, unsafe_allow_html=True)
    
    with analysis_tabs[3]:
        particle_data = results['particulates']
        st.write("**Particulate Analysis:**")
        st.write(f"- Texture Measure: {particle_data['texture_measure']:.3f}")
        st.write(f"- Noise Level: {particle_data['noise_level']:.3f}")
        st.write(f"- Particulate Score: {particle_data['particulate_score']:.3f}")
        
        # Particulate interpretation
        if particle_data['particulate_score'] < 0.3:
            st.success("✅ Low particulate levels")
        elif particle_data['particulate_score'] < 0.6:
            st.warning("⚠️ Moderate particulate levels")
        else:
            st.error("❌ High particulate levels detected")
    
    # Visualization
    st.subheader("📈 Visual Analysis")
    
    analyzer = ImagePollutionAnalyzer()
    fig = analyzer.create_analysis_visualization(results)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Health Precautions
    st.subheader("🏥 Health Precautions & Recommendations")
    
    precautions = results['precautions']
    
    precaution_tabs = st.tabs(["👥 General Public", "⚠️ Sensitive Groups", "🏃 Outdoor Activities", "🛡️ Protective Measures"])
    
    with precaution_tabs[0]:
        st.write("**General Population:**")
        for precaution in precautions['general']:
            st.write(f"• {precaution}")
    
    with precaution_tabs[1]:
        st.write("**Sensitive Groups (Children, Elderly, Respiratory Conditions):**")
        for precaution in precautions['sensitive_groups']:
            st.write(f"• {precaution}")
    
    with precaution_tabs[2]:
        st.write("**Outdoor Activities:**")
        for precaution in precautions['outdoor_activities']:
            st.write(f"• {precaution}")
    
    with precaution_tabs[3]:
        st.write("**Protective Measures:**")
        for precaution in precautions['protective_measures']:
            st.write(f"• {precaution}")
    
    # Export Results
    st.subheader("📤 Export Results")
    
    export_cols = st.columns(3)
    
    with export_cols[0]:
        if st.button("📄 Download JSON Report"):
            json_str = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"air_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with export_cols[1]:
        if st.button("📊 Download CSV Summary"):
            summary_data = {
                'timestamp': [results['timestamp']],
                'location': [location],
                'estimated_aqi': [aqi_estimate['estimated_aqi']],
                'aqi_category': [aqi_estimate['aqi_category']],
                'visibility_score': [results['visibility']['visibility_score']],
                'haze_density': [results['haze']['haze_density']],
                'particulate_score': [results['particulates']['particulate_score']]
            }
            
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"air_quality_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with export_cols[2]:
        if st.button("📋 Copy Analysis Summary"):
            summary_text = f"""
Air Quality Analysis Summary
Location: {location}
Timestamp: {results['timestamp']}
Estimated AQI: {aqi_estimate['estimated_aqi']:.0f} ({aqi_estimate['aqi_category']})
Visibility Score: {results['visibility']['visibility_score']:.2f}
Haze Density: {results['haze']['haze_density']:.2f}
Particulate Level: {results['particulates']['particulate_score']:.2f}
            """
            st.code(summary_text)

if __name__ == "__main__":
    create_image_analysis_ui()