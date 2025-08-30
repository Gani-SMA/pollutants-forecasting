#!/usr/bin/env python3
"""
Simple Database Viewer for Supabase Air Quality Data
"""

import os
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime
import json

def load_credentials():
    """Load credentials from .env file"""
    load_dotenv()
    
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_ANON_KEY')
    
    if not url or not key:
        print("❌ Credentials not found in .env file")
        print()
        print("Make sure you have a .env file with:")
        print("SUPABASE_URL=your_project_url")
        print("SUPABASE_ANON_KEY=your_anon_key")
        return None, None
    
    return url, key

def connect_to_database():
    """Connect to Supabase database"""
    print("🔌 Connecting to Supabase database...")
    
    url, key = load_credentials()
    if not url or not key:
        return None
    
    try:
        supabase = create_client(url, key)
        print("✅ Connected successfully!")
        return supabase
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

def check_tables(supabase):
    """Check if tables exist and show record counts"""
    print("\n📊 Checking Database Tables:")
    print("-" * 30)
    
    tables = [
        'air_quality_analyses',
        'health_recommendations', 
        'analysis_statistics'
    ]
    
    total_records = 0
    
    for table in tables:
        try:
            response = supabase.table(table).select('*').limit(1).execute()
            
            # Get count by fetching all records (simple approach)
            all_response = supabase.table(table).select('id').execute()
            count = len(all_response.data) if all_response.data else 0
            
            print(f"✅ {table}: {count} records")
            total_records += count
            
        except Exception as e:
            print(f"❌ {table}: Error - {str(e)}")
    
    print(f"\n📈 Total Records: {total_records}")
    return total_records > 0

def show_recent_analyses(supabase, limit=5):
    """Show recent air quality analyses"""
    print(f"\n🕒 Recent {limit} Air Quality Analyses:")
    print("-" * 40)
    
    try:
        response = supabase.table('air_quality_analyses').select('*').order('created_at', desc=True).limit(limit).execute()
        
        if not response.data:
            print("📭 No analyses found")
            print("\n💡 To add data:")
            print("1. Run the air quality analyzer app")
            print("2. Upload and analyze an image")
            print("3. Enable database saving")
            return
        
        for i, analysis in enumerate(response.data, 1):
            # Parse timestamp
            created_at = analysis.get('created_at', '')
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    time_str = created_at
            else:
                time_str = 'Unknown'
            
            print(f"\n{i}. Analysis #{analysis.get('id', 'Unknown')}")
            print(f"   📅 Date: {time_str}")
            print(f"   🌡️  AQI: {analysis.get('aqi', 'N/A')} ({analysis.get('category', 'N/A')})")
            print(f"   👁️  Visibility: {int((analysis.get('visibility_score', 0) or 0) * 100)}%")
            print(f"   🌫️  Haze: {int((analysis.get('haze_density', 0) or 0) * 100)}%")
            print(f"   💨 Particulates: {int((analysis.get('particulate_score', 0) or 0) * 100)}%")
            
            # Sky color
            r = analysis.get('sky_r', 0) or 0
            g = analysis.get('sky_g', 0) or 0
            b = analysis.get('sky_b', 0) or 0
            print(f"   🎨 Sky Color: RGB({r}, {g}, {b})")
            
            # Location if available
            if analysis.get('location_name'):
                print(f"   📍 Location: {analysis['location_name']}")
        
    except Exception as e:
        print(f"❌ Error retrieving analyses: {e}")

def show_statistics(supabase):
    """Show database statistics"""
    print("\n📊 Database Statistics:")
    print("-" * 25)
    
    try:
        # Get all analyses for statistics
        response = supabase.table('air_quality_analyses').select('aqi, category, created_at').execute()
        
        if not response.data:
            print("📭 No data for statistics")
            return
        
        analyses = response.data
        
        # Calculate statistics
        aqi_values = [a.get('aqi', 0) for a in analyses if a.get('aqi')]
        
        if aqi_values:
            avg_aqi = sum(aqi_values) / len(aqi_values)
            max_aqi = max(aqi_values)
            min_aqi = min(aqi_values)
            
            print(f"📈 Total Analyses: {len(analyses)}")
            print(f"📊 Average AQI: {avg_aqi:.1f}")
            print(f"📈 Maximum AQI: {max_aqi}")
            print(f"📉 Minimum AQI: {min_aqi}")
            
            # Category distribution
            categories = {}
            for analysis in analyses:
                category = analysis.get('category', 'Unknown')
                categories[category] = categories.get(category, 0) + 1
            
            print(f"\n🎯 Category Distribution:")
            for category, count in categories.items():
                print(f"   {category}: {count}")
        
    except Exception as e:
        print(f"❌ Error calculating statistics: {e}")

def export_data(supabase):
    """Export all data to JSON file"""
    print("\n📤 Exporting Data:")
    print("-" * 18)
    
    try:
        # Get all analyses
        analyses_response = supabase.table('air_quality_analyses').select('*').execute()
        
        # Get all recommendations
        recommendations_response = supabase.table('health_recommendations').select('*').execute()
        
        # Get all statistics
        statistics_response = supabase.table('analysis_statistics').select('*').execute()
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'analyses': analyses_response.data or [],
            'recommendations': recommendations_response.data or [],
            'statistics': statistics_response.data or []
        }
        
        filename = f"air_quality_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✅ Data exported to: {filename}")
        print(f"📊 Analyses: {len(export_data['analyses'])}")
        print(f"🏥 Recommendations: {len(export_data['recommendations'])}")
        print(f"📈 Statistics: {len(export_data['statistics'])}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

def main():
    """Main function"""
    print("🗄️ Supabase Air Quality Database Viewer")
    print("=" * 45)
    
    # Connect to database
    supabase = connect_to_database()
    if not supabase:
        return
    
    # Check tables and show counts
    has_data = check_tables(supabase)
    
    if not has_data:
        print("\n💡 Your database is empty. To add data:")
        print("1. Run: streamlit run modern_air_analyzer.py --server.port 8507")
        print("2. Upload an image and analyze it")
        print("3. Results will be saved automatically")
        return
    
    # Show recent analyses
    show_recent_analyses(supabase, limit=5)
    
    # Show statistics
    show_statistics(supabase)
    
    # Ask about export
    print("\n" + "="*45)
    export_choice = input("Would you like to export all data to JSON? (y/n): ").lower().strip()
    
    if export_choice == 'y':
        export_data(supabase)
    
    print("\n✅ Database check complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Make sure your .env file contains valid Supabase credentials.")