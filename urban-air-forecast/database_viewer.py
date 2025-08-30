import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from supabase_config import SupabaseAirQualityDB
import os

# Page configuration
st.set_page_config(
    page_title="🗄️ Air Quality Database Viewer",
    page_icon="🗄️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }
    
    .stat-label {
        color: #718096;
        font-size: 0.9rem;
        margin: 0;
    }
    
    .analysis-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .aqi-good { border-left: 4px solid #48bb78; }
    .aqi-moderate { border-left: 4px solid #ed8936; }
    .aqi-unhealthy { border-left: 4px solid #e53e3e; }
    .aqi-hazardous { border-left: 4px solid #742a2a; }
</style>
""", unsafe_allow_html=True)

class DatabaseViewer:
    def __init__(self):
        self.db = None
        self.connected = False
        self._init_database()
    
    def _init_database(self):
        """Initialize database connection"""
        try:
            supabase_url = os.getenv('SUPABASE_URL') or st.session_state.get('supabase_url')
            supabase_key = os.getenv('SUPABASE_ANON_KEY') or st.session_state.get('supabase_key')
            
            if supabase_url and supabase_key:
                self.db = SupabaseAirQualityDB(supabase_url, supabase_key)
                self.connected = True
            
        except Exception as e:
            st.error(f"Database connection failed: {str(e)}")
            self.connected = False

def setup_database_connection():
    """Setup database connection"""
    if 'db_viewer' not in st.session_state:
        st.session_state.db_viewer = DatabaseViewer()
    
    if not st.session_state.db_viewer.connected:
        st.error("❌ Database not connected")
        
        with st.expander("🔧 Setup Database Connection"):
            supabase_url = st.text_input("Supabase URL:", type="password")
            supabase_key = st.text_input("Supabase Anon Key:", type="password")
            
            if st.button("Connect"):
                if supabase_url and supabase_key:
                    st.session_state.supabase_url = supabase_url
                    st.session_state.supabase_key = supabase_key
                    st.rerun()
                else:
                    st.error("Please provide both URL and key")
        return False
    
    return True

def display_statistics(db):
    """Display database statistics"""
    st.subheader("📊 Database Statistics")
    
    try:
        # Get statistics for different time periods
        stats_7d = db.get_statistics(days=7)
        stats_30d = db.get_statistics(days=30)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats_7d.get('total_analyses', 0)}</div>
                <div class="stat-label">Analyses (7 days)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats_7d.get('average_aqi', 0)}</div>
                <div class="stat-label">Average AQI (7 days)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats_7d.get('max_aqi', 0)}</div>
                <div class="stat-label">Max AQI (7 days)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats_30d.get('total_analyses', 0)}</div>
                <div class="stat-label">Analyses (30 days)</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Category distribution chart
        if stats_7d.get('category_distribution'):
            st.subheader("📈 Air Quality Distribution (Last 7 Days)")
            
            categories = list(stats_7d['category_distribution'].keys())
            values = list(stats_7d['category_distribution'].values())
            
            fig = px.pie(
                values=values,
                names=categories,
                title="Air Quality Categories",
                color_discrete_map={
                    'good': '#48bb78',
                    'moderate': '#ed8936',
                    'unhealthy': '#e53e3e',
                    'hazardous': '#742a2a'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Daily trend chart
        if stats_7d.get('daily_breakdown'):
            st.subheader("📊 Daily AQI Trend")
            
            daily_data = stats_7d['daily_breakdown']
            df = pd.DataFrame(daily_data)
            
            if not df.empty:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['avg_aqi'],
                    mode='lines+markers',
                    name='Average AQI',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['max_aqi'],
                    mode='lines',
                    name='Max AQI',
                    line=dict(color='#e53e3e', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="Daily AQI Trends",
                    xaxis_title="Date",
                    yaxis_title="AQI",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Failed to load statistics: {str(e)}")

def display_recent_analyses(db):
    """Display recent analyses"""
    st.subheader("🕒 Recent Analyses")
    
    try:
        # Get recent analyses
        recent = db.get_recent_analyses(limit=20)
        
        if recent:
            for analysis in recent:
                # Determine AQI class
                aqi = analysis['aqi']
                if aqi <= 50:
                    aqi_class = 'aqi-good'
                elif aqi <= 100:
                    aqi_class = 'aqi-moderate'
                elif aqi <= 200:
                    aqi_class = 'aqi-unhealthy'
                else:
                    aqi_class = 'aqi-hazardous'
                
                # Format timestamp
                timestamp = datetime.fromisoformat(analysis['created_at'].replace('Z', '+00:00'))
                time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                
                # Display analysis card
                st.markdown(f"""
                <div class="analysis-card {aqi_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>AQI: {analysis['aqi']} ({analysis['category']})</strong><br>
                            <small>{time_str}</small>
                            {f"<br><small>📍 {analysis['location_name']}</small>" if analysis.get('location_name') else ""}
                        </div>
                        <div style="text-align: right;">
                            <small>
                                👁️ {int(analysis['visibility_score'] * 100)}% | 
                                🌫️ {int(analysis['haze_density'] * 100)}% | 
                                💨 {int(analysis['particulate_score'] * 100)}%
                            </small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No analyses found in the database")
    
    except Exception as e:
        st.error(f"Failed to load recent analyses: {str(e)}")

def display_search_interface(db):
    """Display search and filter interface"""
    st.subheader("🔍 Search & Filter")
    
    with st.expander("Search Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            aqi_min = st.number_input("Min AQI:", min_value=0, max_value=500, value=0)
            aqi_max = st.number_input("Max AQI:", min_value=0, max_value=500, value=500)
            category = st.selectbox("Category:", ["All", "Good", "Moderate", "Unhealthy", "Hazardous"])
        
        with col2:
            location = st.text_input("Location (partial match):")
            date_from = st.date_input("From date:", value=datetime.now() - timedelta(days=30))
            date_to = st.date_input("To date:", value=datetime.now())
        
        if st.button("🔍 Search"):
            try:
                # Prepare search parameters
                search_params = {
                    'aqi_min': aqi_min if aqi_min > 0 else None,
                    'aqi_max': aqi_max if aqi_max < 500 else None,
                    'category': category if category != "All" else None,
                    'location': location if location else None,
                    'date_from': date_from.isoformat() if date_from else None,
                    'date_to': date_to.isoformat() if date_to else None,
                    'limit': 50
                }
                
                # Perform search
                results = db.search_analyses(**search_params)
                
                if results:
                    st.success(f"Found {len(results)} results")
                    
                    # Convert to DataFrame for display
                    df_data = []
                    for result in results:
                        df_data.append({
                            'Date': datetime.fromisoformat(result['created_at'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M'),
                            'AQI': result['aqi'],
                            'Category': result['category'],
                            'Location': result.get('location_name', 'N/A'),
                            'Visibility': f"{int(result['visibility_score'] * 100)}%",
                            'Haze': f"{int(result['haze_density'] * 100)}%",
                            'Particulates': f"{int(result['particulate_score'] * 100)}%"
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download search results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Search Results",
                        data=csv,
                        file_name=f"air_quality_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No results found matching your criteria")
                    
            except Exception as e:
                st.error(f"Search failed: {str(e)}")

def display_data_management(db):
    """Display data management options"""
    st.subheader("🛠️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Export Data")
        
        if st.button("📊 Export All Statistics"):
            try:
                stats = db.get_statistics(days=365)  # Get full year stats
                
                stats_json = json.dumps(stats, indent=2, default=str)
                st.download_button(
                    label="📥 Download Statistics JSON",
                    data=stats_json,
                    file_name=f"air_quality_statistics_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
        
        if st.button("📋 Export Recent Analyses"):
            try:
                recent = db.get_recent_analyses(limit=100)
                
                if recent:
                    df_data = []
                    for analysis in recent:
                        df_data.append({
                            'timestamp': analysis['created_at'],
                            'aqi': analysis['aqi'],
                            'category': analysis['category'],
                            'visibility_score': analysis['visibility_score'],
                            'haze_density': analysis['haze_density'],
                            'pollution_tint': analysis['pollution_tint'],
                            'particulate_score': analysis['particulate_score'],
                            'sky_r': analysis['sky_r'],
                            'sky_g': analysis['sky_g'],
                            'sky_b': analysis['sky_b'],
                            'location_name': analysis.get('location_name', ''),
                            'latitude': analysis.get('latitude', ''),
                            'longitude': analysis.get('longitude', '')
                        })
                    
                    df = pd.DataFrame(df_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Analyses CSV",
                        data=csv,
                        file_name=f"air_quality_analyses_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No data to export")
                    
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    with col2:
        st.markdown("### Database Info")
        
        try:
            # Get basic database info
            recent_count = len(db.get_recent_analyses(limit=1000))
            stats = db.get_statistics(days=365)
            
            st.info(f"""
            **Database Summary:**
            - Total analyses: {stats.get('total_analyses', 0)}
            - Date range: Last 365 days
            - Average AQI: {stats.get('average_aqi', 0)}
            - Categories tracked: Good, Moderate, Unhealthy, Hazardous
            """)
            
        except Exception as e:
            st.error(f"Failed to get database info: {str(e)}")

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🗄️ Air Quality Database Viewer</h1>
        <p>View, search, and manage your air quality analysis data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup database connection
    if not setup_database_connection():
        return
    
    db = st.session_state.db_viewer.db
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistics", "🕒 Recent Data", "🔍 Search", "🛠️ Management"])
    
    with tab1:
        display_statistics(db)
    
    with tab2:
        display_recent_analyses(db)
    
    with tab3:
        display_search_interface(db)
    
    with tab4:
        display_data_management(db)

if __name__ == "__main__":
    main()