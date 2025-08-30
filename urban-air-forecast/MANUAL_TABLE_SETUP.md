# 🗄️ Manual Supabase Table Setup Guide

## 📋 How to Create Database Tables in Supabase Dashboard

Since the automatic table creation didn't work, you'll need to create the tables manually in your Supabase dashboard. This is actually very easy!

### Step 1: Open Supabase SQL Editor

1. **Go to your Supabase dashboard:**
   - Visit: https://supabase.com/dashboard
   - Click on your project: `eqfaoewdyvbnozibwjpf`

2. **Open SQL Editor:**
   - In the left sidebar, click **"SQL Editor"**
   - Click **"New Query"**

### Step 2: Create the Tables

Copy and paste each SQL command below into the SQL editor and click **"Run"** for each one:

#### Table 1: Air Quality Analyses
```sql
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
```

#### Table 2: Health Recommendations
```sql
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
```

#### Table 3: Analysis Statistics
```sql
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
```

### Step 3: Verify Tables Were Created

After running all three SQL commands:

1. **Check the Table Editor:**
   - In the left sidebar, click **"Table Editor"**
   - You should see three new tables:
     - `air_quality_analyses`
     - `health_recommendations`
     - `analysis_statistics`

2. **Test the Connection:**
   ```bash
   python simple_db_viewer.py
   ```

## 🎯 What Each Table Does

### `air_quality_analyses`
- **Stores main analysis results** from image processing
- **AQI values, categories, and detailed metrics**
- **Sky color data and pollution indicators**
- **Optional location information**

### `health_recommendations`
- **Stores health advice** for each analysis
- **Immediate actions, protective measures, activity guidelines**
- **Linked to analyses via analysis_id**

### `analysis_statistics`
- **Daily aggregated statistics**
- **Tracks trends over time**
- **Category distribution and averages**

## 🔍 How to Check Your Stored Data

Once tables are created, you have several ways to view your data:

### Method 1: Command Line Viewer
```bash
python simple_db_viewer.py
```

### Method 2: Supabase Dashboard
1. Go to **Table Editor** in your Supabase dashboard
2. Click on any table to view its data
3. Use filters and search to explore

### Method 3: Database Viewer App
```bash
streamlit run database_viewer.py --server.port 8509
```

### Method 4: Export Data
```bash
python simple_db_viewer.py
# Choose export option when prompted
```

## 📊 After Creating Tables

Once your tables are set up:

1. **Run the Air Quality Analyzer:**
   ```bash
   streamlit run modern_air_analyzer.py --server.port 8507
   ```

2. **Upload and analyze an image**

3. **Check that data was saved:**
   ```bash
   python simple_db_viewer.py
   ```

## 🚨 Troubleshooting

### "Table already exists" error
- This is normal and safe - the `IF NOT EXISTS` prevents conflicts

### "Permission denied" error
- Make sure you're using the correct Supabase project
- Verify your API key has the right permissions

### Tables not showing up
- Refresh your browser
- Check the "public" schema in Table Editor
- Wait a few seconds and try again

## 🎉 Success Indicators

You'll know everything is working when:

✅ **Tables appear in Supabase Table Editor**
✅ **`python simple_db_viewer.py` shows "Connected successfully!"**
✅ **Air quality analyzer shows "Database Connected" in sidebar**
✅ **Analysis results are automatically saved**

---

**Ready to create your tables? Just follow the steps above and you'll have a fully functional air quality database!** 🗄️✨