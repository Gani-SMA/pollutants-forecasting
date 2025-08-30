#!/usr/bin/env python3
"""
Test script to demonstrate parameter-specific health recommendations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modern_air_analyzer import ModernAirQualityAnalyzer

def test_recommendations():
    """Test different scenarios for health recommendations"""
    
    analyzer = ModernAirQualityAnalyzer()
    
    print("🧪 Testing Parameter-Specific Health Recommendations")
    print("=" * 60)
    
    # Test Case 1: Good air quality with high visibility
    print("\n📊 Test Case 1: Good Air Quality (AQI: 45)")
    print("-" * 40)
    test_data_1 = {
        'aqi': 45,
        'visibility_score': 0.85,
        'haze_density': 0.2,
        'particulate_score': 0.15,
        'pollution_tint': 0.1
    }
    
    recommendations_1 = analyzer.get_health_recommendations(test_data_1['aqi'], test_data_1)
    print("✅ Immediate Actions:")
    for rec in recommendations_1['immediate']:
        print(f"   • {rec}")
    
    # Test Case 2: Moderate air quality with high haze
    print("\n📊 Test Case 2: Moderate Air Quality with High Haze (AQI: 85)")
    print("-" * 40)
    test_data_2 = {
        'aqi': 85,
        'visibility_score': 0.4,
        'haze_density': 0.8,
        'particulate_score': 0.5,
        'pollution_tint': 0.4
    }
    
    recommendations_2 = analyzer.get_health_recommendations(test_data_2['aqi'], test_data_2)
    print("⚠️ Immediate Actions:")
    for rec in recommendations_2['immediate']:
        print(f"   • {rec}")
    print("\n🛡️ Protective Measures:")
    for rec in recommendations_2['protective']:
        print(f"   • {rec}")
    
    # Test Case 3: Unhealthy air quality with poor visibility and high particulates
    print("\n📊 Test Case 3: Unhealthy Air Quality with Poor Visibility (AQI: 150)")
    print("-" * 40)
    test_data_3 = {
        'aqi': 150,
        'visibility_score': 0.25,
        'haze_density': 0.9,
        'particulate_score': 0.75,
        'pollution_tint': 0.6
    }
    
    recommendations_3 = analyzer.get_health_recommendations(test_data_3['aqi'], test_data_3)
    print("🚨 Immediate Actions:")
    for rec in recommendations_3['immediate']:
        print(f"   • {rec}")
    print("\n🛡️ Protective Measures:")
    for rec in recommendations_3['protective']:
        print(f"   • {rec}")
    print("\n🏃‍♂️ Activity Guidelines:")
    for rec in recommendations_3['activities']:
        print(f"   • {rec}")
    
    # Test Case 4: Hazardous conditions
    print("\n📊 Test Case 4: Hazardous Air Quality (AQI: 350)")
    print("-" * 40)
    test_data_4 = {
        'aqi': 350,
        'visibility_score': 0.1,
        'haze_density': 0.95,
        'particulate_score': 0.9,
        'pollution_tint': 0.8
    }
    
    recommendations_4 = analyzer.get_health_recommendations(test_data_4['aqi'], test_data_4)
    print("🚨 Immediate Actions:")
    for rec in recommendations_4['immediate']:
        print(f"   • {rec}")
    
    print("\n" + "=" * 60)
    print("✅ Test completed! Health recommendations are now parameter-specific.")
    print("📱 Open the Streamlit app to see these recommendations in action:")
    print("   streamlit run modern_air_analyzer.py --server.port 8508")

if __name__ == "__main__":
    test_recommendations()