# 💻 How to Enter Supabase Credentials in Terminal

## Step-by-Step Terminal Login Process

### 1. Open Terminal and Navigate to Project
```bash
cd urban-air-forecast
```

### 2. Run the Credential Helper
```bash
python credential_helper.py
```

### 3. Follow the Terminal Prompts

#### **Prompt 1: Account Check**
```
Do you already have a Supabase account? (y/n):
```
**What to type:**
- Type `n` and press Enter if you need to create an account
- Type `y` and press Enter if you already have an account

#### **Prompt 2: Project URL**
```
🌐 Paste your Project URL:
```
**What to type:**
- Go to your Supabase dashboard
- Copy the URL that looks like: `https://abcdefghijk.supabase.co`
- Paste it in the terminal and press Enter

**Example:**
```
🌐 Paste your Project URL: https://xyzabc123def.supabase.co
```

#### **Prompt 3: API Key**
```
🔑 Paste your anon/public API key:
```
**What to type:**
- In Supabase dashboard, go to Settings → API
- Copy the "anon public" key (long string starting with "eyJ...")
- Paste it in the terminal and press Enter

**Example:**
```
🔑 Paste your anon/public API key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRlc3QiLCJyb2xlIjoiYW5vbiIsImlhdCI6MTY0MjQ5ODEwMCwiZXhwIjoxOTU4MDc0MTAwfQ.abc123def456ghi789
```

### 4. Confirmation Messages
After entering credentials, you'll see:
```
✅ Project URL looks good!
✅ API key looks good!
💾 Saving credentials...
✅ Credentials saved to .env file
🔌 Testing database connection...
✅ Database connection successful!
```

## 🔍 Where to Find Your Credentials

### Getting Your Project URL:
1. **Login to Supabase:** https://supabase.com/dashboard
2. **Select your project**
3. **Look at the browser URL** or go to Settings → General
4. **Copy the Project URL:** `https://[your-project-id].supabase.co`

### Getting Your API Key:
1. **In your Supabase project dashboard**
2. **Click "Settings"** (gear icon) in left sidebar
3. **Click "API"** from settings menu
4. **Copy the "anon public" key** (NOT the service_role key)

## 📝 Copy-Paste Tips for Terminal

### Windows (Command Prompt/PowerShell):
- **Copy:** Ctrl+C (from browser)
- **Paste:** Right-click in terminal OR Ctrl+V (in newer terminals)

### Windows (Git Bash):
- **Copy:** Ctrl+C (from browser)
- **Paste:** Shift+Insert OR Right-click → Paste

### Mac Terminal:
- **Copy:** Cmd+C (from browser)
- **Paste:** Cmd+V (in terminal)

### Linux Terminal:
- **Copy:** Ctrl+C (from browser)
- **Paste:** Ctrl+Shift+V OR Right-click → Paste

## ⚠️ Common Issues and Solutions

### Issue: "Nothing happens when I paste"
**Solution:**
- Make sure you're using the correct paste shortcut for your terminal
- Try right-clicking and selecting "Paste"
- Some terminals require Shift+Insert

### Issue: "Invalid API Key"
**Solution:**
- Make sure you copied the **anon public** key, not service_role
- Check for extra spaces at the beginning or end
- Copy the entire key (it's very long)

### Issue: "Project URL not found"
**Solution:**
- Ensure the URL starts with `https://`
- Make sure it ends with `.supabase.co`
- Verify your project is fully created (wait 2-3 minutes after creation)

### Issue: "Permission denied" or "Access denied"
**Solution:**
- Make sure you're in the correct directory: `cd urban-air-forecast`
- Check that you have write permissions in the folder
- Try running as administrator if needed

## 🎯 Complete Example Session

Here's what a complete terminal session looks like:

```bash
$ cd urban-air-forecast
$ python credential_helper.py

🚀 Supabase Credential Setup Helper
==================================================
This tool will help you create and configure Supabase credentials
for your Air Quality Image Analyzer database.

Do you already have a Supabase account? (y/n): y
✅ Great! Let's get your credentials.

Step 2: Get Your Credentials
------------------------------
🔑 Getting Your Credentials:

1. 🏠 In your Supabase project dashboard:
   • Click 'Settings' (gear icon) in the left sidebar
   • Click 'API' from the settings menu

2. 📋 Copy these two values:
   • Project URL (looks like: https://abc123.supabase.co)
   • anon/public key (long string starting with 'eyJ...')

⚠️  IMPORTANT: Use the 'anon public' key, NOT the 'service_role' key!

Press Enter when you can see your API settings page...

Step 3: Enter Credentials
-------------------------
📝 Enter Your Credentials:

🌐 Paste your Project URL: https://myproject123.supabase.co
✅ Project URL looks good!

🔑 Paste your anon/public API key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
✅ API key looks good!

Step 4: Save & Test
--------------------
💾 Saving credentials...
✅ Credentials saved to .env file
✅ Added .env to .gitignore for security
🔌 Testing database connection...
✅ Database connection successful!
📋 Setting up database tables...
✅ Database tables created successfully!

🎉 Setup Complete!
==============================

Your Supabase database is ready! Here's what you can do next:

1. 📸 Run the Air Quality Analyzer with database:
   streamlit run air_analyzer_with_db.py --server.port 8508
```

## 🚀 Quick Start Commands

After entering credentials successfully:

```bash
# Run the main analyzer with database
streamlit run air_analyzer_with_db.py --server.port 8508

# Or run the database viewer
streamlit run database_viewer.py --server.port 8509
```

## 🔒 Security Notes

- Your credentials are saved to a `.env` file
- This file is automatically added to `.gitignore`
- Never share your `.env` file or commit it to version control
- The setup uses the safe "anon public" key, not the sensitive service_role key

**Ready to enter your credentials? Just run `python credential_helper.py` and follow the prompts!** 🔑✨