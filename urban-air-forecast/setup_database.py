#!/usr/bin/env python3
"""
Supabase Database Setup Script for Air Quality Analyzer
"""

import os
import sys
from pathlib import Path
from supabase_config import SupabaseAirQualityDB, setup_environment_variables, test_connection

def main():
    print("🗄️ Air Quality Analyzer - Supabase Database Setup")
    print("=" * 60)
    print()
    
    # Check if supabase is installed
    try:
        import supabase
        print("✅ Supabase library is installed")
    except ImportError:
        print("❌ Supabase library not found. Installing...")
        os.system(f"{sys.executable} -m pip install supabase python-dotenv")
        print("✅ Supabase library installed")
    
    print()
    
    # Check for existing environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        print("🔧 Setting up Supabase credentials...")
        print()
        print("You'll need to:")
        print("1. Create a Supabase account at https://supabase.com")
        print("2. Create a new project")
        print("3. Get your project URL and anon key from Settings > API")
        print()
        
        supabase_url, supabase_key = setup_environment_variables()
    else:
        print("✅ Found existing Supabase credentials")
    
    print()
    
    # Test connection
    print("🔌 Testing database connection...")
    if test_connection(supabase_url, supabase_key):
        print("✅ Connection successful!")
        
        # Create database instance
        print()
        print("📋 Creating database tables...")
        db = SupabaseAirQualityDB(supabase_url, supabase_key)
        
        # Create tables
        if db.create_tables():
            print("✅ Database tables created successfully!")
            
            # Test insert
            print()
            print("🧪 Testing database operations...")
            
            # Sample data for testing
            sample_analysis = {
                'success': True,
                'aqi': 85,
                'category': 'Moderate',
                'visibility_score': 0.75,
                'haze_density': 0.35,
                'pollution_tint': 0.12,
                'particulate_score': 0.28,
                'sky_rgb': [180, 190, 200],
                'pollution_index': 0.24,
                'analysis_time': '14:30:25'
            }
            
            sample_recommendations = {
                'immediate': ['Test recommendation 1', 'Test recommendation 2'],
                'protective': ['Test protective measure 1'],
                'activities': ['Test activity guideline 1']
            }
            
            sample_metadata = {
                'name': 'test_image.jpg',
                'size_kb': 250,
                'dimensions': '800x600'
            }
            
            # Try to save test data
            analysis_id = db.save_analysis_result(
                sample_analysis,
                sample_recommendations,
                sample_metadata,
                user_id='setup_test'
            )
            
            if analysis_id:
                print(f"✅ Test data saved successfully (ID: {analysis_id})")
                
                # Try to retrieve test data
                retrieved = db.get_analysis_by_id(analysis_id)
                if retrieved:
                    print("✅ Test data retrieved successfully")
                    
                    # Clean up test data
                    if db.delete_analysis(analysis_id):
                        print("✅ Test data cleaned up")
                    
                    print()
                    print("🎉 Database setup completed successfully!")
                    print()
                    print("Next steps:")
                    print("1. Run the air quality analyzer: streamlit run air_analyzer_with_db.py")
                    print("2. Upload images and analyze air quality")
                    print("3. Results will be automatically saved to your Supabase database")
                    print()
                    print("Database features:")
                    print("• Automatic saving of analysis results")
                    print("• Health recommendations storage")
                    print("• Historical data and statistics")
                    print("• Search and filter capabilities")
                    print("• Export functionality")
                    
                else:
                    print("❌ Failed to retrieve test data")
            else:
                print("❌ Failed to save test data")
        else:
            print("❌ Failed to create database tables")
    else:
        print("❌ Connection failed. Please check your credentials.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)