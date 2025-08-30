# 🗄️ Supabase Database Integration for Air Quality Analyzer

## Overview

This integration adds powerful database capabilities to the Air Quality Image Analyzer, allowing you to:

- **Store all analysis results** automatically
- **Track historical air quality data** over time
- **Generate statistics and trends** from your data
- **Search and filter** past analyses
- **Export data** in multiple formats
- **View comprehensive dashboards** of air quality metrics

## 🚀 Quick Setup

### 1. Create Supabase Account

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up for a free account
3. Create a new project
4. Wait for the project to be ready (usually 1-2 minutes)

### 2. Get Your Credentials

1. In your Supabase dashboard, go to **Settings** → **API**
2. Copy your **Project URL**
3. Copy your **anon/public key**

### 3. Install Dependencies

```bash
cd urban-air-forecast
pip install -r requirements_supabase.txt
```

### 4. Run Database Setup

```bash
python setup_database.py
```

This will:
- Prompt you for your Supabase credentials
- Create a `.env` file with your credentials
- Test the database connection
- Create all necessary tables
- Run a test to ensure everything works

### 5. Run the Application

```bash
streamlit run air_analyzer_with_db.py --server.port 8508
```

## 📊 Database Schema

The system creates three main tables:

### `air_quality_analyses`
Stores the main analysis results:
- **Analysis Results**: AQI, category, visibility, haze, particulates
- **Sky Color Data**: RGB values and pollution tint
- **Image Metadata**: Name, size, dimensions
- **Location Data**: Optional location name and coordinates
- **Timestamps**: Creation and update times

### `health_recommendations`
Stores health recommendations for each analysis:
- **Immediate Actions**: What to do right now
- **Protective Measures**: Equipment and precautions
- **Activity Guidelines**: Exercise and outdoor activity advice
- **Severity Levels**: Risk categorization

### `analysis_statistics`
Stores daily aggregated statistics:
- **Daily Totals**: Number of analyses per day
- **AQI Statistics**: Average, min, max AQI values
- **Category Counts**: Distribution of air quality categories
- **Trends**: Historical data for trend analysis

## 🎯 Features

### Automatic Data Storage
- Every analysis is automatically saved to the database
- Includes all metrics, recommendations, and metadata
- Optional location data for geographic tracking

### Real-time Statistics
- Live dashboard showing recent trends
- Category distribution charts
- Daily AQI trend graphs
- Historical comparisons

### Advanced Search
- Filter by AQI range
- Search by location
- Date range filtering
- Category-based filtering
- Export search results

### Data Export
- JSON reports with complete analysis data
- CSV exports for spreadsheet analysis
- Statistical summaries
- Historical trend data

## 🔧 Configuration

### Environment Variables

Create a `.env` file in your project directory:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

### Manual Configuration

You can also set credentials directly in the app:
1. Run the application
2. Go to the sidebar "Database Connection" section
3. Enter your Supabase URL and key
4. Click "Connect to Database"

## 📱 Using the Database Features

### Main Analyzer App (`air_analyzer_with_db.py`)

1. **Automatic Saving**: Enable "Save results to database" in settings
2. **Location Tracking**: Optionally add location information
3. **View Statistics**: Click "Show Database Stats" in sidebar
4. **Connection Status**: Green indicator shows database connection

### Database Viewer (`database_viewer.py`)

Run the database viewer for advanced data management:

```bash
streamlit run database_viewer.py --server.port 8509
```

Features:
- **Statistics Dashboard**: Comprehensive analytics
- **Recent Data**: View latest analyses
- **Search Interface**: Advanced filtering options
- **Data Management**: Export and maintenance tools

## 📊 Analytics & Insights

### Key Metrics Tracked
- **AQI Trends**: Daily, weekly, monthly averages
- **Category Distribution**: Good vs. Moderate vs. Unhealthy vs. Hazardous
- **Location Patterns**: Geographic air quality variations
- **Seasonal Trends**: Long-term air quality changes

### Visualizations
- **Pie Charts**: Air quality category distribution
- **Line Graphs**: AQI trends over time
- **Statistics Cards**: Key performance indicators
- **Historical Comparisons**: Period-over-period analysis

## 🔍 Search & Filter Capabilities

### Available Filters
- **AQI Range**: Min/max AQI values
- **Category**: Good, Moderate, Unhealthy, Hazardous
- **Location**: Partial text matching
- **Date Range**: From/to date filtering
- **Time Period**: Last 7 days, 30 days, custom ranges

### Export Options
- **CSV**: Spreadsheet-compatible format
- **JSON**: Complete data with metadata
- **Summary Reports**: Human-readable analysis

## 🛠️ Maintenance & Management

### Database Maintenance
- Automatic statistics updates
- Data integrity checks
- Performance optimization
- Backup recommendations

### Data Cleanup
- Remove test data
- Archive old analyses
- Optimize storage usage
- Maintain data quality

## 🔒 Security & Privacy

### Data Protection
- All data stored in your private Supabase instance
- Row-level security available
- API key authentication
- HTTPS encryption for all communications

### Privacy Considerations
- Location data is optional
- No personal information required
- User IDs are optional and can be anonymous
- Full control over your data

## 🚨 Troubleshooting

### Common Issues

#### Connection Failed
- Verify Supabase URL and key are correct
- Check internet connection
- Ensure Supabase project is active

#### Tables Not Created
- Run `python setup_database.py` again
- Check Supabase dashboard for table creation
- Verify database permissions

#### Data Not Saving
- Check database connection status
- Verify "Save to database" is enabled
- Look for error messages in the app

#### Performance Issues
- Large datasets may slow queries
- Consider data archiving for old records
- Use filters to limit result sets

### Getting Help

1. **Check Logs**: Look for error messages in the Streamlit app
2. **Verify Setup**: Re-run the setup script
3. **Test Connection**: Use the connection test feature
4. **Supabase Dashboard**: Check your project status

## 📈 Advanced Usage

### Custom Queries
Access the database directly for custom analysis:

```python
from supabase_config import SupabaseAirQualityDB

db = SupabaseAirQualityDB()
custom_data = db.supabase.table('air_quality_analyses').select('*').gte('aqi', 100).execute()
```

### API Integration
Use Supabase's REST API for external integrations:
- Mobile app development
- Third-party analytics tools
- Automated reporting systems
- Data science workflows

### Scaling Considerations
- Free tier: 500MB database, 2GB bandwidth
- Paid tiers available for larger datasets
- Consider data archiving strategies
- Monitor usage in Supabase dashboard

## 🎉 Benefits

### For Individuals
- **Track Personal Exposure**: Monitor air quality in your area
- **Health Planning**: Make informed decisions about outdoor activities
- **Trend Analysis**: Understand air quality patterns over time

### For Organizations
- **Environmental Monitoring**: Track air quality across multiple locations
- **Health & Safety**: Protect employees and visitors
- **Compliance Reporting**: Generate reports for regulatory requirements
- **Research Data**: Contribute to air quality research

### For Researchers
- **Data Collection**: Systematic air quality measurements
- **Statistical Analysis**: Large datasets for research
- **Collaboration**: Share data with research teams
- **Publication**: Export data for academic papers

## 🔄 Updates & Maintenance

### Regular Updates
- Keep Supabase client library updated
- Monitor for new features and improvements
- Review security best practices
- Backup important data regularly

### Feature Roadmap
- Real-time notifications for poor air quality
- Mobile app integration
- Advanced machine learning analytics
- Integration with weather data
- Community data sharing features

---

## 📞 Support

For technical support:
1. Check this documentation first
2. Review Supabase documentation
3. Check GitHub issues
4. Contact support team

**Happy analyzing! 🌍📊**