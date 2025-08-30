#!/usr/bin/env python3
"""
Create Supabase Tables for Air Quality Database
"""

import os
from dotenv import load_dotenv
from supabase import create_client

def load_credentials():
    """Load credentials from .env file"""
    load_dotenv()
    
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_ANON_KEY')
    
    if not url or not key:
        print("❌ Credentials not found in .env file")
        return None, None
    
    return url, key

def create_tables():
    """Create database tables"""
    print("🗄️ Creating Supabase Tables for Air Quality Database")
    print("=" * 55)
    
    # Connect to database
    url, key = load_credentials()
    if not url or not key:
        return False
    
    try:
        supabase = create_client(url, key)
        print("✅ Connected to Supabase")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # SQL to create tables
    create_analyses_table = """
    CREATE TABLE IF NOT EXISTS air_quality_analyses (
        id SERIAL PRIMARY KEY,
        analysis_id UUID DEFAULT gen_random_uuid() UNIQUE,
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        
        -- Analysis Results
        aqi INTEGER NOT NULL,
        category VARCHAR(50) NOT NULL,
        visibility_score DECIMAL(5,3),
        haze_density DECIMAL(5,3),
        pollution_tint DECIMAL(5,3),
        particulate_score DECIMAL(5,3),
        pollution_index DECIMAL(5,3),
        
        -- Sky Color Data
        sky_r INTEGER,
        sky_g INTEGER,
        sky_b INTEGER,
        
        -- Image Metadata
        image_name VARCHAR(255),
        image_size_kb INTEGER,
        image_dimensions VARCHAR(50),
        analysis_time VARCHAR(20),
        
        -- Location Data (optional)
        location_name VARCHAR(255),
        latitude DECIMAL(10,8),
        longitude DECIMAL(11,8),
        
        -- Additional metadata
        user_id VARCHAR(255),
        device_info TEXT,
        notes TEXT,
        
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    
    create_recommendations_table = """
    CREATE TABLE IF NOT EXISTS health_recommendations (
        id SERIAL PRIMARY KEY,
        analysis_id UUID,
        
        -- Recommendation Categories
        immediate_actions TEXT[],
        protective_measures TEXT[],
        activity_guidelines TEXT[],
        
        -- Recommendation metadata
        aqi_range VARCHAR(20),
        severity_level VARCHAR(20),
        
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    
    create_statistics_table = """
    CREATE TABLE IF NOT EXISTS analysis_statistics (
        id SERIAL PRIMARY KEY,
        date DATE DEFAULT CURRENT_DATE,
        
        -- Daily statistics
        total_analyses INTEGER DEFAULT 0,
        avg_aqi DECIMAL(6,2),
        max_aqi INTEGER,
        min_aqi INTEGER,
        
        -- Category counts
        good_count INTEGER DEFAULT 0,
        moderate_count INTEGER DEFAULT 0,
        unhealthy_count INTEGER DEFAULT 0,
        hazardous_count INTEGER DEFAULT 0,
        
        -- Location statistics (if available)
        unique_locations INTEGER DEFAULT 0,
        
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        
        UNIQUE(date)
    );
    """
    
    # Execute table creation
    tables = [
        ("air_quality_analyses", create_analyses_table),
        ("health_recommendations", create_recommendations_table),
        ("analysis_statistics", create_statistics_table)
    ]
    
    print("\n📋 Creating tables...")
    
    for table_name, sql in tables:
        try:
            # Use RPC to execute raw SQL
            result = supabase.rpc('exec_sql', {'sql': sql}).execute()
            print(f"✅ Created table: {table_name}")
        except Exception as e:
            print(f"❌ Failed to create {table_name}: {e}")
            
            # Try alternative method - direct SQL execution
            try:
                # For Supabase, we need to use the SQL editor or create tables via dashboard
                print(f"⚠️  Please create table {table_name} manually in Supabase dashboard")
                print(f"   Go to: https://supabase.com/dashboard/project/{url.split('//')[1].split('.')[0]}/editor")
                print(f"   Run this SQL:")
                print(f"   {sql}")
                print()
            except:
                pass
    
    print("\n🎉 Table creation process completed!")
    print("\nIf tables weren't created automatically, please:")
    print("1. Go to your Supabase dashboard")
    print("2. Navigate to SQL Editor")
    print("3. Run the SQL commands shown above")
    
    return True

def test_tables():
    """Test if tables were created successfully"""
    print("\n🔍 Testing table creation...")
    
    url, key = load_credentials()
    if not url or not key:
        return False
    
    try:
        supabase = create_client(url, key)
        
        # Test each table
        tables = ['air_quality_analyses', 'health_recommendations', 'analysis_statistics']
        
        for table in tables:
            try:
                response = supabase.table(table).select('*').limit(1).execute()
                print(f"✅ Table {table} is accessible")
            except Exception as e:
                print(f"❌ Table {table} not accessible: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    success = create_tables()
    
    if success:
        test_tables()
        
        print("\n📝 Next Steps:")
        print("1. Run: streamlit run modern_air_analyzer.py --server.port 8507")
        print("2. Upload and analyze an image")
        print("3. Check stored data with: python simple_db_viewer.py")

if __name__ == "__main__":
    main()