#!/usr/bin/env python3
"""
Simple Supabase Credential Entry
"""

import os
import sys

def main():
    print("🔑 Supabase Credential Entry")
    print("=" * 40)
    print()
    
    print("📋 You need two things from your Supabase dashboard:")
    print("1. Project URL (from Settings → General)")
    print("2. anon public key (from Settings → API)")
    print()
    
    # Get Project URL
    print("Step 1: Enter Project URL")
    print("-" * 25)
    print("Example: https://abcdefghijk.supabase.co")
    
    while True:
        url = input("🌐 Project URL: ").strip()
        
        if not url:
            print("❌ Please enter a URL")
            continue
            
        if not url.startswith("https://"):
            print("❌ URL should start with https://")
            continue
            
        if ".supabase.co" not in url:
            print("❌ URL should contain .supabase.co")
            continue
            
        print("✅ URL looks good!")
        break
    
    print()
    
    # Get API Key
    print("Step 2: Enter API Key")
    print("-" * 20)
    print("⚠️  Use the 'anon public' key, NOT 'service_role'!")
    print("Example: eyJhbGciOiJIUzI1NiIsInR5cCI6...")
    
    while True:
        key = input("🔑 API Key: ").strip()
        
        if not key:
            print("❌ Please enter an API key")
            continue
            
        if len(key) < 50:
            print("❌ API key seems too short")
            continue
            
        print("✅ API key looks good!")
        break
    
    print()
    
    # Save credentials
    print("💾 Saving credentials...")
    
    env_content = f"""SUPABASE_URL={url}
SUPABASE_ANON_KEY={key}
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Credentials saved to .env file")
        
        # Test connection
        print("🔌 Testing connection...")
        
        os.environ['SUPABASE_URL'] = url
        os.environ['SUPABASE_ANON_KEY'] = key
        
        try:
            from supabase import create_client
            supabase = create_client(url, key)
            
            # Simple test query
            response = supabase.table('_realtime_schema').select('*').limit(1).execute()
            print("✅ Connection successful!")
            
        except ImportError:
            print("📦 Installing Supabase library...")
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'supabase'])
            print("✅ Supabase installed. Please run the script again.")
            return
            
        except Exception as e:
            print(f"⚠️  Connection test: {str(e)}")
            print("✅ Credentials saved anyway. You can test later.")
        
        print()
        print("🎉 Setup complete!")
        print()
        print("Next steps:")
        print("1. Run: streamlit run air_analyzer_with_db.py --server.port 8508")
        print("2. Enable 'Save to database' in the app")
        print("3. Start analyzing images!")
        
    except Exception as e:
        print(f"❌ Error saving credentials: {e}")

if __name__ == "__main__":
    main()