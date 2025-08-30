#!/usr/bin/env python3
"""
Check Stored Data in Supabase Database
"""

import os
from datetime import datetime
import json
from supabase_config import SupabaseAirQualityDB

def print_header():
    """Print header"""
    print("🗄️ Database Data Checker")
    print("=" * 40)
    print()

def check_connection():
    """Check database connection"""
    try:
        db = SupabaseAirQualityDB()
        print("✅ Database connection successful!")
        return db
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print()
        print("Make sure you have:")
        print("1. ✅ Supabase credentials in .env file")
        print("2. ✅ Internet connection")
        print("3. ✅ Valid Supabase project")
        return None

def show_table_counts(db):
    """Show count of records in each table"""
    print("📊 Table Record Counts:")
    print("-" * 25)
    
    try:
        # Count analyses
        analyses_response = db.supabase.table('air_quality_analyses').select('id', count='exact').execute()
        analyses_count = analyses_response.count if hasattr(analyses_response, 'count') else len(analyses_response.data)
        
        # Count recommendations
        recommendations_response = db.supabase.table('health_recommendations').select('id', count='exact').execute()
        recommendations_count = recommendations_response.count if hasattr(recommendations_response, 'count') else len(recommendations_response.data)
        
        # Count statistics
        statistics_response = db.supabase.table('analysis_statistics').select('id', count='exact').execute()
        statistics_count = statistics_response.count if hasattr(statistics_response, 'count') else len(statistics_response.data)
        
        print(f"📋 Air Quality Analyses: {analyses_count}")
        print(f"🏥 Health Recommendations: {recommendations_count}")
        print(f"📈 Daily Statistics: {statistics_count}")
        print()
        
        return analyses_count > 0
        
    except Exception as e:
        print(f"❌ Error checking table counts: {e}")
        return False

def show_recent_analyses(db, limit=5):
    """Show recent analyses"""
    print(f"🕒 Recent {limit} Analyses:")
    print("-" * 30)
    
    try:
        recent = db.get_recent_analyses(limit=limit)
        
        if not recent:
            print("📭 No analyses found in database")
            print()
            print("To add data:")
            print("1. Run: streamlit run modern_air_analyzer.py --server.port 8507")
            print("2. Upload an image and analyze it")
            print("3. Enable 'Save to database' if available")
            return
        
        for i, analysis in enumerate(recent, 1):
            timestamp = datetime.fromisoformat(analysis['created_at'].replace('Z', '+00:00'))
            time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"{i}. 📅 {time_str}")
            print(f"   🌡️  AQI: {analysis['aqi']} ({analysis['category']})")
            print(f"   👁️  Visibility: {int(analysis['visibility_score'] * 100)}%")
            print(f"   🌫️  Haze: {int(analysis['haze_density'] * 100)}%")
            print(f"   💨 Particulates: {int(analysis['particulate_score'] * 100)}%")
            
            if analysis.get('location_name'):
                print(f"   📍 Location: {analysis['location_name']}")
            
            print()
            
    except Exception as e:
        print(f"❌ Error retrieving recent analyses: {e}")

def show_statistics(db):
    """Show database statistics"""
    print("📊 Database Statistics:")
    print("-" * 25)
    
    try:
        stats = db.get_statistics(days=30)
        
        if stats.get('total_analyses', 0) == 0:
            print("📭 No statistics available yet")
            return
        
        print(f"📈 Total Analyses (30 days): {stats.get('total_analyses', 0)}")
        print(f"📊 Average AQI: {stats.get('average_aqi', 0):.1f}")
        print(f"📈 Max AQI: {stats.get('max_aqi', 0)}")
        print(f"📉 Min AQI: {stats.get('min_aqi', 0)}")
        print()
        
        # Category distribution
        categories = stats.get('category_distribution', {})
        if categories:
            print("🎯 Air Quality Distribution:")
            for category, count in categories.items():
                if count > 0:
                    print(f"   {category.title()}: {count}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error retrieving statistics: {e}")

def show_detailed_analysis(db, analysis_id=None):
    """Show detailed analysis"""
    try:
        if not analysis_id:
            # Get the most recent analysis
            recent = db.get_recent_analyses(limit=1)
            if not recent:
                print("📭 No analyses found")
                return
            analysis_id = recent[0]['analysis_id']
        
        print(f"🔍 Detailed Analysis (ID: {analysis_id[:8]}...):")
        print("-" * 45)
        
        result = db.get_analysis_by_id(analysis_id)
        
        if not result:
            print("❌ Analysis not found")
            return
        
        analysis = result['analysis']
        recommendations = result.get('recommendations', {})
        
        # Analysis details
        timestamp = datetime.fromisoformat(analysis['created_at'].replace('Z', '+00:00'))
        print(f"📅 Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌡️  AQI: {analysis['aqi']} ({analysis['category']})")
        print()
        
        print("📊 Detailed Metrics:")
        print(f"   👁️  Visibility Score: {analysis['visibility_score']:.3f}")
        print(f"   🌫️  Haze Density: {analysis['haze_density']:.3f}")
        print(f"   🌤️  Pollution Tint: {analysis['pollution_tint']:.3f}")
        print(f"   💨 Particulate Score: {analysis['particulate_score']:.3f}")
        print(f"   🎨 Sky Color: RGB({analysis['sky_r']}, {analysis['sky_g']}, {analysis['sky_b']})")
        print()
        
        if analysis.get('location_name'):
            print(f"📍 Location: {analysis['location_name']}")
            if analysis.get('latitude') and analysis.get('longitude'):
                print(f"🗺️  Coordinates: {analysis['latitude']}, {analysis['longitude']}")
            print()
        
        # Health recommendations
        if recommendations:
            print("🏥 Health Recommendations:")
            
            if recommendations.get('immediate_actions'):
                print("   🚨 Immediate Actions:")
                for action in recommendations['immediate_actions']:
                    print(f"      • {action}")
                print()
            
            if recommendations.get('protective_measures'):
                print("   🛡️  Protective Measures:")
                for measure in recommendations['protective_measures']:
                    print(f"      • {measure}")
                print()
            
            if recommendations.get('activity_guidelines'):
                print("   🏃‍♂️ Activity Guidelines:")
                for guideline in recommendations['activity_guidelines']:
                    print(f"      • {guideline}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error retrieving detailed analysis: {e}")

def export_all_data(db):
    """Export all data to JSON file"""
    print("📤 Exporting All Data:")
    print("-" * 22)
    
    try:
        # Get all analyses
        all_analyses = db.get_recent_analyses(limit=1000)  # Get up to 1000 records
        
        if not all_analyses:
            print("📭 No data to export")
            return
        
        # Get statistics
        stats = db.get_statistics(days=365)
        
        # Prepare export data
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_analyses': len(all_analyses),
            'statistics': stats,
            'analyses': all_analyses
        }
        
        # Save to file
        filename = f"air_quality_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✅ Data exported to: {filename}")
        print(f"📊 Exported {len(all_analyses)} analyses")
        print()
        
    except Exception as e:
        print(f"❌ Error exporting data: {e}")

def interactive_menu(db):
    """Interactive menu for data exploration"""
    while True:
        print("🔍 Data Explorer Menu:")
        print("-" * 20)
        print("1. 📊 Show table counts")
        print("2. 🕒 Show recent analyses")
        print("3. 📈 Show statistics")
        print("4. 🔍 Show detailed analysis")
        print("5. 📤 Export all data")
        print("6. 🚪 Exit")
        print()
        
        choice = input("Choose an option (1-6): ").strip()
        print()
        
        if choice == '1':
            show_table_counts(db)
        elif choice == '2':
            limit = input("How many recent analyses to show? (default 5): ").strip()
            limit = int(limit) if limit.isdigit() else 5
            show_recent_analyses(db, limit)
        elif choice == '3':
            show_statistics(db)
        elif choice == '4':
            show_detailed_analysis(db)
        elif choice == '5':
            export_all_data(db)
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("Press Enter to continue...")
        print()

def main():
    """Main function"""
    print_header()
    
    # Check connection
    db = check_connection()
    if not db:
        return
    
    print()
    
    # Quick overview
    has_data = show_table_counts(db)
    
    if not has_data:
        print("💡 To add data to your database:")
        print("1. Run: streamlit run modern_air_analyzer.py --server.port 8507")
        print("2. Upload an image and analyze it")
        print("3. The results will be automatically saved")
        print()
        return
    
    # Show recent data
    show_recent_analyses(db, limit=3)
    
    # Show statistics
    show_statistics(db)
    
    # Ask if user wants interactive menu
    explore = input("Would you like to explore the data interactively? (y/n): ").lower()
    if explore == 'y':
        print()
        interactive_menu(db)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure your Supabase credentials are set up correctly.")