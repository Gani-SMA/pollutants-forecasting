# 📸 Air Quality Image Analyzer

## Overview
The Air Quality Image Analyzer is an advanced computer vision application that estimates air quality parameters from photographs. Using sophisticated image processing algorithms, it analyzes visual pollution indicators and provides health recommendations.

## 🌟 Key Features

### 📸 **Multiple Input Methods**
- **📁 File Upload**: Upload images from your device
- **📷 Camera Capture**: Real-time photo capture using device camera
- **🌐 URL Input**: Analyze images from web URLs
- **🎯 Sample Images**: Test with pre-loaded examples

### 🔬 **Advanced Analysis Capabilities**
1. **👁️ Visibility Analysis**
   - Atmospheric clarity measurement
   - Contrast and edge detection
   - Detail visibility scoring

2. **🌫️ Haze/Smog Detection**
   - Color saturation analysis
   - Brightness uniformity measurement
   - Haze density calculation

3. **🌤️ Sky Color Analysis**
   - RGB color extraction from sky region
   - Pollution tint detection (brown/yellow indicates pollution)
   - Gray level analysis for desaturation

4. **💨 Particulate Detection**
   - Noise variance analysis for visible particles
   - Texture measurement for dust/particulates
   - Combined particulate scoring

### 📊 **AQI Estimation**
- Weighted combination of all visual indicators
- Mapping to standard AQI scale (0-500)
- Confidence scoring based on indicator agreement
- Category classification (Good → Hazardous)

### 🏥 **Health Recommendations**
- **🚨 Immediate Actions** based on AQI level
- **🛡️ Protective Measures** (masks, air purifiers)
- **🏃‍♂️ Activity Modifications** (outdoor exercise limits)
- **👥 Vulnerable Group Precautions** (children, elderly, respiratory conditions)

### 📈 **Visualizations**
- **📊 Radar Chart** showing pollution indicators
- **🎨 Sky Color Display** with RGB values
- **📋 Metric Cards** with interpretations
- **🚦 AQI Status** with color-coded categories

### 📤 **Export Options**
- **📄 JSON Report** - Complete technical analysis
- **📝 Summary Report** - Human-readable markdown
- **📊 CSV Data** - Structured data export

## 🚀 Quick Start

### Installation & Setup
```bash
# Navigate to project directory
cd urban-air-forecast

# Install requirements (automatic)
python run_image_analyzer.py
```

### Running the Application
```bash
# Method 1: Using launcher script
python run_image_analyzer.py

# Method 2: Direct streamlit command
streamlit run image_capture_analyzer.py --server.port 8503
```

### Access the Application
Open your browser and navigate to: `http://localhost:8503`

## 📋 Requirements

### Python Packages
- `streamlit>=1.28.0`
- `opencv-python>=4.8.0`
- `Pillow>=10.0.0`
- `numpy>=1.24.0`
- `plotly>=5.15.0`
- `requests>=2.28.0`
- `pandas>=1.5.0`

### System Requirements
- Python 3.8 or higher
- Camera access (for camera capture feature)
- Internet connection (for URL image loading)

## 🎯 How to Use

### 1. **Choose Input Method**
- Select from sidebar: Upload, Camera, URL, or Sample images
- Follow the specific instructions for each method

### 2. **Upload/Capture Image**
- **Best Results**: Outdoor images with visible sky and distant objects
- **Lighting**: Good natural lighting preferred
- **Focus**: Clear, sharp images work best

### 3. **Analyze Image**
- Click "🔬 Analyze Air Quality" button
- Or enable "Auto-analyze on upload" for automatic processing

### 4. **Review Results**
- **AQI Estimate**: Main air quality index with confidence level
- **Detailed Metrics**: Individual pollution indicators
- **Radar Chart**: Visual representation of all indicators

### 5. **Health Recommendations**
- **Immediate Actions**: What to do right now
- **Protective Measures**: Equipment and precautions
- **Activity Modifications**: Changes to daily activities
- **Vulnerable Groups**: Special considerations

### 6. **Export Data**
- Choose from JSON, Summary, or CSV formats
- Download for documentation or further analysis

## 🔬 Technical Details

### Analysis Algorithms

#### Visibility Analysis
```python
# Contrast measurement
contrast = np.std(gray_image) / 255.0

# Edge detection for detail visibility
edges = cv2.Canny(gray_image, 50, 150)
edge_density = np.sum(edges > 0) / total_pixels

# Combined visibility score
visibility_score = (contrast * 0.7 + edge_density * 0.3) * 2
```

#### Haze Detection
```python
# Color saturation analysis
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
avg_saturation = np.mean(hsv[:, :, 1]) / 255.0

# Brightness uniformity
brightness_std = np.std(hsv[:, :, 2]) / 255.0

# Haze density calculation
haze_density = 1.0 - (avg_saturation * 0.6 + brightness_std * 0.4)
```

#### AQI Estimation Formula
```python
pollution_index = (
    (1 - visibility_score) * 0.25 +    # Poor visibility = pollution
    haze_density * 0.30 +              # Haze indicates pollution
    pollution_tint * 0.20 +            # Color tint from pollution
    particulate_score * 0.25           # Visible particles
)

estimated_aqi = pollution_index * 350  # Scale to AQI range
```

### AQI Categories
| Category | Range | Color | Health Impact |
|----------|-------|-------|---------------|
| Good | 0-50 | Green | Minimal impact |
| Moderate | 51-100 | Yellow | Acceptable for most |
| Unhealthy for Sensitive Groups | 101-150 | Orange | Sensitive individuals at risk |
| Unhealthy | 151-200 | Red | Everyone may experience effects |
| Very Unhealthy | 201-300 | Purple | Health alert conditions |
| Hazardous | 301-500 | Maroon | Emergency conditions |

## 🎨 UI Features

### Responsive Design
- **Wide Layout**: Optimized for desktop and tablet viewing
- **Mobile Friendly**: Responsive design adapts to smaller screens
- **Dark/Light Mode**: Follows system preferences

### Interactive Elements
- **Real-time Analysis**: Immediate feedback on image upload
- **Progress Indicators**: Visual feedback during processing
- **Error Handling**: Clear error messages and recovery suggestions

### Accessibility
- **Color-blind Friendly**: Uses patterns and text in addition to colors
- **Screen Reader Compatible**: Proper ARIA labels and descriptions
- **Keyboard Navigation**: Full keyboard accessibility support

## 🔧 Customization

### Analysis Parameters
You can modify the analysis weights in the `ImageAirQualityAnalyzer` class:

```python
pollution_index = (
    (1 - visibility['visibility_score']) * 0.25 +  # Adjust weight
    haze['haze_density'] * 0.30 +                  # Adjust weight
    sky['pollution_tint'] * 0.20 +                 # Adjust weight
    particulates['particulate_score'] * 0.25       # Adjust weight
)
```

### Health Recommendations
Modify the `get_health_recommendations()` method to customize advice for different AQI levels.

### UI Styling
Update the CSS in the `st.markdown()` sections to change colors, fonts, and layout.

## 🐛 Troubleshooting

### Common Issues

#### Camera Not Working
- **Check Permissions**: Ensure browser has camera access
- **HTTPS Required**: Camera capture requires HTTPS in production
- **Fallback**: Use file upload as alternative

#### Poor Analysis Results
- **Image Quality**: Ensure clear, well-lit outdoor images
- **Sky Visibility**: Include sky and distant objects in frame
- **Lighting Conditions**: Avoid extreme lighting (too dark/bright)

#### Performance Issues
- **Image Size**: Large images may take longer to process
- **Browser Memory**: Refresh page if experiencing slowdowns
- **Network**: URL loading requires stable internet connection

### Error Messages
- **"Analysis Failed"**: Check image format and quality
- **"URL Loading Error"**: Verify URL is accessible and points to image
- **"Camera Access Denied"**: Grant camera permissions in browser

## 📊 Accuracy & Limitations

### Accuracy Factors
- **✅ Good Conditions**: Clear outdoor images with visible sky
- **✅ Optimal Lighting**: Natural daylight conditions
- **✅ Stable Weather**: Non-extreme weather conditions

### Limitations
- **⚠️ Indoor Images**: Not designed for indoor air quality
- **⚠️ Night Images**: Reduced accuracy in low light
- **⚠️ Weather Effects**: Rain/snow may affect readings
- **⚠️ Camera Quality**: Better cameras provide more accurate results

### Validation
- Compare results with official AQI measurements when available
- Use multiple images from same location for consistency
- Consider environmental factors that may affect visual assessment

## 🤝 Contributing

### Development Setup
```bash
git clone <repository>
cd urban-air-forecast
pip install -r requirements.txt
```

### Adding Features
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

### Reporting Issues
- Use GitHub Issues for bug reports
- Include image samples and error messages
- Specify browser and system information

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision algorithms
- Streamlit team for the excellent web framework
- Air quality research community for validation methods
- Beta testers for feedback and improvements

---

**📸 Start analyzing air quality from images today!** 🌍