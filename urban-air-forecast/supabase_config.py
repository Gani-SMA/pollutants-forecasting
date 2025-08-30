"""
Supabase Database Configuration and Setup
"""

import os
from supabase import create_client, Client
from datetime import datetime
import json
from typing import Dict, Any, Optional, List

class SupabaseAirQualityDB:
    """Supabase database client for air quality analysis results"""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Initialize Supabase client
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anon key
        """
        # Use environment variables or provided values
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_ANON_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and key must be provided either as parameters or environment variables")
        
        # Create Supabase client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
    
    def create_tables(self):
        """Create necessary tables for air quality analysis"""
        
        # SQL to create air_quality_analyses table
        create_analyses_table = """
        CREATE TABLE IF NOT EXISTS air_quality_analyses (
            id SERIAL PRIMARY KEY,
            analysis_id UUID DEFAULT gen_random_uuid(),
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
        
        # SQL to create health_recommendations table
        create_recommendations_table = """
        CREATE TABLE IF NOT EXISTS health_recommendations (
            id SERIAL PRIMARY KEY,
            analysis_id UUID REFERENCES air_quality_analyses(analysis_id),
            
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
        
        # SQL to create analysis_statistics table for aggregated data
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
        
        try:
            # Execute table creation queries
            self.supabase.rpc('exec_sql', {'sql': create_analyses_table}).execute()
            self.supabase.rpc('exec_sql', {'sql': create_recommendations_table}).execute()
            self.supabase.rpc('exec_sql', {'sql': create_statistics_table}).execute()
            
            print("✅ Database tables created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error creating tables: {str(e)}")
            return False
    
    def save_analysis_result(self, 
                           analysis_results: Dict[str, Any], 
                           health_recommendations: Dict[str, List[str]],
                           image_metadata: Dict[str, Any] = None,
                           location_data: Dict[str, Any] = None,
                           user_id: str = None) -> Optional[str]:
        """
        Save air quality analysis results to database
        
        Args:
            analysis_results: Results from image analysis
            health_recommendations: Health recommendations based on AQI
            image_metadata: Optional image metadata
            location_data: Optional location information
            user_id: Optional user identifier
            
        Returns:
            analysis_id if successful, None if failed
        """
        try:
            # Prepare analysis data
            analysis_data = {
                'aqi': analysis_results['aqi'],
                'category': analysis_results['category'],
                'visibility_score': analysis_results['visibility_score'],
                'haze_density': analysis_results['haze_density'],
                'pollution_tint': analysis_results['pollution_tint'],
                'particulate_score': analysis_results['particulate_score'],
                'pollution_index': analysis_results['pollution_index'],
                'sky_r': analysis_results['sky_rgb'][0],
                'sky_g': analysis_results['sky_rgb'][1],
                'sky_b': analysis_results['sky_rgb'][2],
                'analysis_time': analysis_results['analysis_time']
            }
            
            # Add image metadata if provided
            if image_metadata:
                analysis_data.update({
                    'image_name': image_metadata.get('name'),
                    'image_size_kb': image_metadata.get('size_kb'),
                    'image_dimensions': image_metadata.get('dimensions')
                })
            
            # Add location data if provided
            if location_data:
                analysis_data.update({
                    'location_name': location_data.get('name'),
                    'latitude': location_data.get('latitude'),
                    'longitude': location_data.get('longitude')
                })
            
            # Add user ID if provided
            if user_id:
                analysis_data['user_id'] = user_id
            
            # Insert analysis data
            analysis_response = self.supabase.table('air_quality_analyses').insert(analysis_data).execute()
            
            if not analysis_response.data:
                raise Exception("Failed to insert analysis data")
            
            analysis_id = analysis_response.data[0]['analysis_id']
            
            # Prepare recommendations data
            recommendations_data = {
                'analysis_id': analysis_id,
                'immediate_actions': health_recommendations['immediate'],
                'protective_measures': health_recommendations['protective'],
                'activity_guidelines': health_recommendations['activities'],
                'aqi_range': self._get_aqi_range(analysis_results['aqi']),
                'severity_level': analysis_results['category']
            }
            
            # Insert recommendations data
            recommendations_response = self.supabase.table('health_recommendations').insert(recommendations_data).execute()
            
            if not recommendations_response.data:
                print("⚠️ Warning: Failed to save health recommendations")
            
            # Update daily statistics
            self._update_daily_statistics(analysis_results['aqi'], analysis_results['category'])
            
            print(f"✅ Analysis saved successfully with ID: {analysis_id}")
            return str(analysis_id)
            
        except Exception as e:
            print(f"❌ Error saving analysis: {str(e)}")
            return None
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis results by ID"""
        try:
            # Get analysis data
            analysis_response = self.supabase.table('air_quality_analyses').select('*').eq('analysis_id', analysis_id).execute()
            
            if not analysis_response.data:
                return None
            
            analysis = analysis_response.data[0]
            
            # Get recommendations data
            recommendations_response = self.supabase.table('health_recommendations').select('*').eq('analysis_id', analysis_id).execute()
            
            recommendations = recommendations_response.data[0] if recommendations_response.data else {}
            
            return {
                'analysis': analysis,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"❌ Error retrieving analysis: {str(e)}")
            return None
    
    def get_recent_analyses(self, limit: int = 10, user_id: str = None) -> List[Dict[str, Any]]:
        """Get recent analysis results"""
        try:
            query = self.supabase.table('air_quality_analyses').select('*').order('created_at', desc=True).limit(limit)
            
            if user_id:
                query = query.eq('user_id', user_id)
            
            response = query.execute()
            return response.data if response.data else []
            
        except Exception as e:
            print(f"❌ Error retrieving recent analyses: {str(e)}")
            return []
    
    def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get analysis statistics for the last N days"""
        try:
            # Get daily statistics
            stats_response = self.supabase.table('analysis_statistics').select('*').order('date', desc=True).limit(days).execute()
            
            daily_stats = stats_response.data if stats_response.data else []
            
            # Calculate overall statistics
            if daily_stats:
                total_analyses = sum(stat['total_analyses'] for stat in daily_stats)
                avg_aqi = sum(stat['avg_aqi'] * stat['total_analyses'] for stat in daily_stats) / total_analyses if total_analyses > 0 else 0
                max_aqi = max(stat['max_aqi'] for stat in daily_stats)
                min_aqi = min(stat['min_aqi'] for stat in daily_stats)
                
                category_totals = {
                    'good': sum(stat['good_count'] for stat in daily_stats),
                    'moderate': sum(stat['moderate_count'] for stat in daily_stats),
                    'unhealthy': sum(stat['unhealthy_count'] for stat in daily_stats),
                    'hazardous': sum(stat['hazardous_count'] for stat in daily_stats)
                }
            else:
                total_analyses = avg_aqi = max_aqi = min_aqi = 0
                category_totals = {'good': 0, 'moderate': 0, 'unhealthy': 0, 'hazardous': 0}
            
            return {
                'period_days': days,
                'total_analyses': total_analyses,
                'average_aqi': round(avg_aqi, 2),
                'max_aqi': max_aqi,
                'min_aqi': min_aqi,
                'category_distribution': category_totals,
                'daily_breakdown': daily_stats
            }
            
        except Exception as e:
            print(f"❌ Error retrieving statistics: {str(e)}")
            return {}
    
    def search_analyses(self, 
                       aqi_min: int = None, 
                       aqi_max: int = None, 
                       category: str = None,
                       location: str = None,
                       date_from: str = None,
                       date_to: str = None,
                       limit: int = 50) -> List[Dict[str, Any]]:
        """Search analyses with filters"""
        try:
            query = self.supabase.table('air_quality_analyses').select('*')
            
            if aqi_min is not None:
                query = query.gte('aqi', aqi_min)
            if aqi_max is not None:
                query = query.lte('aqi', aqi_max)
            if category:
                query = query.eq('category', category)
            if location:
                query = query.ilike('location_name', f'%{location}%')
            if date_from:
                query = query.gte('created_at', date_from)
            if date_to:
                query = query.lte('created_at', date_to)
            
            response = query.order('created_at', desc=True).limit(limit).execute()
            return response.data if response.data else []
            
        except Exception as e:
            print(f"❌ Error searching analyses: {str(e)}")
            return []
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete an analysis and its recommendations"""
        try:
            # Delete recommendations first (foreign key constraint)
            self.supabase.table('health_recommendations').delete().eq('analysis_id', analysis_id).execute()
            
            # Delete analysis
            response = self.supabase.table('air_quality_analyses').delete().eq('analysis_id', analysis_id).execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            print(f"❌ Error deleting analysis: {str(e)}")
            return False
    
    def _get_aqi_range(self, aqi: int) -> str:
        """Get AQI range string"""
        if aqi <= 50:
            return "0-50"
        elif aqi <= 100:
            return "51-100"
        elif aqi <= 200:
            return "101-200"
        else:
            return "201-500"
    
    def _update_daily_statistics(self, aqi: int, category: str):
        """Update daily statistics"""
        try:
            today = datetime.now().date().isoformat()
            
            # Get existing statistics for today
            existing_stats = self.supabase.table('analysis_statistics').select('*').eq('date', today).execute()
            
            if existing_stats.data:
                # Update existing record
                stats = existing_stats.data[0]
                new_total = stats['total_analyses'] + 1
                new_avg_aqi = ((stats['avg_aqi'] * stats['total_analyses']) + aqi) / new_total
                
                update_data = {
                    'total_analyses': new_total,
                    'avg_aqi': round(new_avg_aqi, 2),
                    'max_aqi': max(stats['max_aqi'], aqi),
                    'min_aqi': min(stats['min_aqi'], aqi),
                    'updated_at': datetime.now().isoformat()
                }
                
                # Update category counts
                category_field = f"{category.lower().replace(' ', '_').replace('for_sensitive_groups', '')}_count"
                if category_field in ['good_count', 'moderate_count', 'unhealthy_count', 'hazardous_count']:
                    update_data[category_field] = stats.get(category_field, 0) + 1
                
                self.supabase.table('analysis_statistics').update(update_data).eq('date', today).execute()
            else:
                # Create new record
                new_stats = {
                    'date': today,
                    'total_analyses': 1,
                    'avg_aqi': aqi,
                    'max_aqi': aqi,
                    'min_aqi': aqi,
                    'good_count': 1 if category == 'Good' else 0,
                    'moderate_count': 1 if category == 'Moderate' else 0,
                    'unhealthy_count': 1 if 'Unhealthy' in category else 0,
                    'hazardous_count': 1 if category == 'Hazardous' else 0
                }
                
                self.supabase.table('analysis_statistics').insert(new_stats).execute()
                
        except Exception as e:
            print(f"⚠️ Warning: Failed to update statistics: {str(e)}")

# Environment setup helper
def setup_environment_variables():
    """Helper function to set up environment variables"""
    print("🔧 Supabase Environment Setup")
    print("=" * 50)
    print("Please provide your Supabase credentials:")
    print()
    
    supabase_url = input("Enter your Supabase URL: ").strip()
    supabase_key = input("Enter your Supabase anon key: ").strip()
    
    # Create .env file
    env_content = f"""# Supabase Configuration
SUPABASE_URL={supabase_url}
SUPABASE_ANON_KEY={supabase_key}
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Environment variables saved to .env file")
    print("📝 Make sure to add .env to your .gitignore file!")
    
    return supabase_url, supabase_key

# Test connection function
def test_connection(supabase_url: str = None, supabase_key: str = None):
    """Test Supabase connection"""
    try:
        db = SupabaseAirQualityDB(supabase_url, supabase_key)
        
        # Test basic connection
        response = db.supabase.table('air_quality_analyses').select('count').execute()
        
        print("✅ Supabase connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Setup script
    print("🗄️ Supabase Air Quality Database Setup")
    print("=" * 50)
    
    # Check if environment variables exist
    if not os.getenv('SUPABASE_URL') or not os.getenv('SUPABASE_ANON_KEY'):
        print("Environment variables not found. Setting up...")
        supabase_url, supabase_key = setup_environment_variables()
    else:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')
        print("Using existing environment variables...")
    
    # Test connection
    if test_connection(supabase_url, supabase_key):
        # Create database instance and tables
        db = SupabaseAirQualityDB(supabase_url, supabase_key)
        db.create_tables()
        print("🎉 Database setup complete!")
    else:
        print("❌ Setup failed. Please check your credentials.")