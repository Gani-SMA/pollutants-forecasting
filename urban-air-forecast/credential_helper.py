#!/usr/bin/env python3
"""
Interactive Supabase Credential Setup Helper
"""

import os
import sys
import webbrowser
from pathlib import Path
import time

def print_header():
    """Print welcome header"""
    print("🚀 Supabase Credential Setup Helper")
    print("=" * 50)
    print("This tool will help you create and configure Supabase credentials")
    print("for your Air Quality Image Analyzer database.")
    print()

def open_supabase_signup():
    """Open Supabase signup page"""
    print("📱 Opening Supabase signup page...")
    try:
        webbrowser.open("https://supabase.com/dashboard/sign-up")
        print("✅ Supabase signup page opened in your browser")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")
        print("Please manually visit: https://supabase.com/dashboard/sign-up")
    print()

def guide_account_creation():
    """Guide through account creation"""
    print("📋 Account Creation Steps:")
    print()
    print("1. 🔐 Sign up using one of these options:")
    print("   • GitHub (Recommended)")
    print("   • Google")
    print("   • Email & Password")
    print()
    print("2. ✉️ Verify your email (if using email signup)")
    print()
    print("3. 🎯 Create a new project:")
    print("   • Project Name: air-quality-analyzer")
    print("   • Database Password: [Create a strong password]")
    print("   • Region: [Choose closest to you]")
    print("   • Plan: Free")
    print()
    
    input("Press Enter when you've completed account creation...")
    print()

def guide_credential_extraction():
    """Guide through getting credentials"""
    print("🔑 Getting Your Credentials:")
    print()
    print("1. 🏠 In your Supabase project dashboard:")
    print("   • Click 'Settings' (gear icon) in the left sidebar")
    print("   • Click 'API' from the settings menu")
    print()
    print("2. 📋 Copy these two values:")
    print("   • Project URL (looks like: https://abc123.supabase.co)")
    print("   • anon/public key (long string starting with 'eyJ...')")
    print()
    print("⚠️  IMPORTANT: Use the 'anon public' key, NOT the 'service_role' key!")
    print()
    
    input("Press Enter when you can see your API settings page...")
    print()

def collect_credentials():
    """Collect credentials from user"""
    print("📝 Enter Your Credentials:")
    print()
    
    # Get Project URL
    while True:
        project_url = input("🌐 Paste your Project URL: ").strip()
        
        if not project_url:
            print("❌ Project URL cannot be empty")
            continue
            
        if not project_url.startswith("https://"):
            print("❌ Project URL should start with 'https://'")
            continue
            
        if ".supabase.co" not in project_url:
            print("❌ Project URL should contain '.supabase.co'")
            continue
            
        break
    
    print("✅ Project URL looks good!")
    print()
    
    # Get API Key
    while True:
        api_key = input("🔑 Paste your anon/public API key: ").strip()
        
        if not api_key:
            print("❌ API key cannot be empty")
            continue
            
        if len(api_key) < 100:
            print("❌ API key seems too short. Make sure you copied the full key.")
            continue
            
        if not api_key.startswith("eyJ"):
            print("⚠️  Warning: API key should typically start with 'eyJ'")
            confirm = input("Are you sure this is correct? (y/n): ").lower()
            if confirm != 'y':
                continue
            
        break
    
    print("✅ API key looks good!")
    print()
    
    return project_url, api_key

def save_credentials(project_url, api_key):
    """Save credentials to .env file"""
    print("💾 Saving credentials...")
    
    env_content = f"""# Supabase Configuration for Air Quality Analyzer
# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}

SUPABASE_URL={project_url}
SUPABASE_ANON_KEY={api_key}

# Security Note:
# - Keep this file private
# - Add .env to your .gitignore
# - Never share your service_role key
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ Credentials saved to .env file")
        
        # Create/update .gitignore
        gitignore_path = Path('.gitignore')
        gitignore_content = ""
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
        
        if '.env' not in gitignore_content:
            with open(gitignore_path, 'a') as f:
                f.write('\n# Environment variables\n.env\n*.env\n.env.local\n')
            print("✅ Added .env to .gitignore for security")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving credentials: {e}")
        return False

def test_connection(project_url, api_key):
    """Test the database connection"""
    print("🔌 Testing database connection...")
    
    try:
        # Set environment variables temporarily
        os.environ['SUPABASE_URL'] = project_url
        os.environ['SUPABASE_ANON_KEY'] = api_key
        
        # Import and test
        from supabase_config import SupabaseAirQualityDB
        
        db = SupabaseAirQualityDB(project_url, api_key)
        
        # Try a simple query to test connection
        response = db.supabase.table('air_quality_analyses').select('count').execute()
        
        print("✅ Database connection successful!")
        return True
        
    except ImportError:
        print("⚠️  Supabase library not installed. Installing now...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'supabase', 'python-dotenv'])
            print("✅ Supabase library installed successfully")
            return test_connection(project_url, api_key)  # Retry
        except Exception as e:
            print(f"❌ Failed to install Supabase library: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        print()
        print("Common issues:")
        print("• Check that your Project URL is correct")
        print("• Verify you're using the anon/public key (not service_role)")
        print("• Make sure your Supabase project is fully created")
        print("• Check your internet connection")
        return False

def setup_database_tables():
    """Set up database tables"""
    print("📋 Setting up database tables...")
    
    try:
        from supabase_config import SupabaseAirQualityDB
        
        db = SupabaseAirQualityDB()
        
        if db.create_tables():
            print("✅ Database tables created successfully!")
            return True
        else:
            print("❌ Failed to create database tables")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up tables: {e}")
        return False

def show_next_steps():
    """Show what to do next"""
    print("🎉 Setup Complete!")
    print("=" * 30)
    print()
    print("Your Supabase database is ready! Here's what you can do next:")
    print()
    print("1. 📸 Run the Air Quality Analyzer with database:")
    print("   streamlit run air_analyzer_with_db.py --server.port 8508")
    print()
    print("2. 📊 View your database dashboard:")
    print("   streamlit run database_viewer.py --server.port 8509")
    print()
    print("3. 🔍 In the analyzer app:")
    print("   • Enable 'Save results to database' in settings")
    print("   • Upload images and analyze air quality")
    print("   • View statistics in the sidebar")
    print()
    print("4. 📈 Track your data over time:")
    print("   • All analyses are automatically saved")
    print("   • View trends and statistics")
    print("   • Export data for further analysis")
    print()
    print("🔒 Security reminder:")
    print("• Your .env file contains sensitive credentials")
    print("• Never share or commit this file to version control")
    print("• The .gitignore has been updated to protect it")
    print()

def main():
    """Main setup flow"""
    print_header()
    
    # Check if credentials already exist
    if os.path.exists('.env'):
        print("📁 Found existing .env file")
        overwrite = input("Do you want to create new credentials? (y/n): ").lower()
        if overwrite != 'y':
            print("👋 Setup cancelled. Using existing credentials.")
            return
        print()
    
    # Step 1: Guide to create account
    print("Step 1: Create Supabase Account")
    print("-" * 35)
    
    has_account = input("Do you already have a Supabase account? (y/n): ").lower()
    
    if has_account != 'y':
        open_supabase_signup()
        guide_account_creation()
    else:
        print("✅ Great! Let's get your credentials.")
        print()
    
    # Step 2: Guide to get credentials
    print("Step 2: Get Your Credentials")
    print("-" * 30)
    guide_credential_extraction()
    
    # Step 3: Collect credentials
    print("Step 3: Enter Credentials")
    print("-" * 25)
    project_url, api_key = collect_credentials()
    
    # Step 4: Save credentials
    print("Step 4: Save & Test")
    print("-" * 20)
    if not save_credentials(project_url, api_key):
        print("❌ Setup failed. Please try again.")
        return
    
    # Step 5: Test connection
    if not test_connection(project_url, api_key):
        print("❌ Connection test failed. Please check your credentials.")
        return
    
    # Step 6: Set up database
    if not setup_database_tables():
        print("❌ Database setup failed. You may need to set up tables manually.")
        return
    
    # Step 7: Show next steps
    print()
    show_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please try running the setup again or check the documentation.")
        sys.exit(1)