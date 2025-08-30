# 🚀 How to Create Supabase Credentials

## Step-by-Step Guide to Set Up Supabase for Air Quality Analyzer

### Step 1: Create Supabase Account

1. **Go to Supabase Website**
   - Open your browser and visit: [https://supabase.com](https://supabase.com)
   - Click on **"Start your project"** or **"Sign Up"**

2. **Sign Up Options**
   - **GitHub Account** (Recommended): Click "Continue with GitHub"
   - **Google Account**: Click "Continue with Google"
   - **Email/Password**: Enter your email and create a password

3. **Verify Your Account**
   - Check your email for verification link (if using email signup)
   - Click the verification link to activate your account

### Step 2: Create a New Project

1. **Access Dashboard**
   - After signing in, you'll see the Supabase dashboard
   - Click **"New Project"** or **"Create a new project"**

2. **Choose Organization**
   - Select your personal organization (usually your username)
   - Or create a new organization if needed

3. **Project Configuration**
   ```
   Project Name: air-quality-analyzer
   Database Password: [Create a strong password - SAVE THIS!]
   Region: Choose closest to your location
   Pricing Plan: Free (perfect for getting started)
   ```

4. **Create Project**
   - Click **"Create new project"**
   - Wait 1-2 minutes for project setup to complete
   - You'll see a progress indicator

### Step 3: Get Your Credentials

1. **Navigate to API Settings**
   - In your project dashboard, click **"Settings"** (gear icon) in the left sidebar
   - Click **"API"** from the settings menu

2. **Copy Your Credentials**
   
   **Project URL:**
   ```
   https://[your-project-id].supabase.co
   ```
   
   **API Keys:**
   - **anon/public key**: This is what you'll use for the app
   - **service_role key**: Keep this secret (don't use in the app)

3. **Save Your Credentials**
   - Copy the **Project URL**
   - Copy the **anon public key** (NOT the service_role key)
   - Save these in a secure location

### Step 4: Test Your Setup

1. **Run the Setup Script**
   ```bash
   cd urban-air-forecast
   python setup_database.py
   ```

2. **Enter Your Credentials**
   - When prompted, paste your **Project URL**
   - When prompted, paste your **anon public key**

3. **Verify Connection**
   - The script will test your connection
   - Create necessary database tables
   - Run a test data insertion

### Step 5: Start Using the Database

1. **Run the Enhanced Analyzer**
   ```bash
   streamlit run air_analyzer_with_db.py --server.port 8508
   ```

2. **Enable Database Saving**
   - In the app sidebar, you'll see "✅ Database Connected"
   - Check "Save results to database"
   - Start analyzing images!

## 🔒 Security Best Practices

### What to Keep Secret
- ❌ **Never share your service_role key**
- ❌ **Don't commit credentials to Git**
- ✅ **Only use the anon/public key in your app**
- ✅ **Store credentials in .env file**

### Credential Storage
The setup script will create a `.env` file:
```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
```

### Git Security
Add to your `.gitignore`:
```
.env
*.env
.env.local
```

## 📊 Free Tier Limits

Supabase free tier includes:
- **Database**: 500MB storage
- **Bandwidth**: 2GB per month
- **API Requests**: 50,000 per month
- **Authentication**: 50,000 monthly active users

Perfect for personal air quality monitoring!

## 🆘 Troubleshooting

### Common Issues

**"Invalid API Key"**
- Make sure you're using the **anon/public** key, not service_role
- Check for extra spaces when copying
- Regenerate key if needed

**"Project Not Found"**
- Verify the Project URL is correct
- Make sure project is fully created (wait 2-3 minutes)
- Check project status in Supabase dashboard

**"Connection Timeout"**
- Check your internet connection
- Try again in a few minutes
- Verify Supabase service status

### Getting Help

1. **Supabase Documentation**: [https://supabase.com/docs](https://supabase.com/docs)
2. **Community Support**: [https://github.com/supabase/supabase/discussions](https://github.com/supabase/supabase/discussions)
3. **Status Page**: [https://status.supabase.com](https://status.supabase.com)

## 🎯 Next Steps

After setting up credentials:

1. **✅ Test the connection** with setup script
2. **📸 Analyze some images** with database saving enabled
3. **📊 View your data** in the database viewer
4. **📈 Track trends** over time
5. **📤 Export your data** for analysis

## 💡 Pro Tips

### Organization
- Use descriptive project names
- Keep credentials in a password manager
- Document your setup for team members

### Usage
- Start with the free tier
- Monitor your usage in the dashboard
- Upgrade when you need more resources

### Development
- Use separate projects for development/production
- Test with sample data first
- Set up proper backup procedures

---

**Ready to create your Supabase account? Follow the steps above and you'll be storing air quality data in minutes!** 🌍📊