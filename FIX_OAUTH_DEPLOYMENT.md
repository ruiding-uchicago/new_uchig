# OAuth Authentication Fix for DigitalOcean Deployment

## Problem Summary
Your Django app on DigitalOcean App Platform has intermittent OAuth login failures due to:
1. Session state not persisting between OAuth redirect requests
2. Multiple container instances not sharing session data
3. Incorrect HTTPS redirect configuration
4. SQLite database not suitable for containerized deployments

## Critical Fixes Applied

### 1. **Session Configuration** (MOST IMPORTANT)
Changed in `settings.py`:
```python
# OLD (BROKEN):
SOCIAL_AUTH_REDIRECT_IS_HTTPS = False  # ❌ This was the main issue!
SESSION_COOKIE_DOMAIN = 'clownfish-app-3wxq3.ondigitalocean.app'
SESSION_ENGINE = 'django.contrib.sessions.backends.cached_db'

# NEW (FIXED):
SOCIAL_AUTH_REDIRECT_IS_HTTPS = True  # ✅ MUST be True for HTTPS OAuth
SESSION_COOKIE_DOMAIN = None  # ✅ Let Django handle automatically
SESSION_ENGINE = 'django.contrib.sessions.backends.db'  # ✅ Use DB only
SESSION_SAVE_EVERY_REQUEST = True  # ✅ Persist on every request
```

### 2. **Cookie Settings**
```python
SESSION_COOKIE_SAMESITE = 'Lax'  # Changed from None
CSRF_COOKIE_SAMESITE = 'Lax'  # Match session cookie
USE_X_FORWARDED_HOST = True  # Trust proxy headers
USE_X_FORWARDED_PORT = True  # Trust proxy ports
```

## Deployment Steps

### Step 1: Add PostgreSQL Database (REQUIRED)
SQLite doesn't work properly with multiple containers. You MUST add a PostgreSQL database:

1. In DigitalOcean App Platform dashboard:
   - Click "Add Resource" → "Database"
   - Choose "Dev Database" ($7/month) or "Basic" ($15/month)
   - Select PostgreSQL
   - Click "Add Database"

2. The DATABASE_URL will be automatically injected as an environment variable

### Step 2: Add Environment Variables
In App Settings → App-Level Environment Variables, add:

```bash
DJANGO_SETTINGS_MODULE=myportal.settings
DATABASE_URL=${db.DATABASE_URL}  # Auto-populated by DO
REDIS_URL=${redis.REDIS_URL}  # Optional, if you add Redis
```

### Step 3: Update Build Commands
In App Settings → Components → new-uchig → Build Command:

```bash
python manage.py collectstatic --noinput && python manage.py migrate
```

### Step 4: Deploy Changes

1. Commit and push the changes:
```bash
git add .
git commit -m "Fix OAuth authentication for DigitalOcean deployment"
git push origin main
```

2. The app will auto-deploy, or you can trigger manually in the dashboard

### Step 5: Verify Database Migration
After deployment, run in the Console:
```bash
python manage.py migrate
python manage.py createsuperuser  # Create admin user if needed
```

## Testing the Fix

1. **Clear your browser cookies** for the domain
2. Visit: https://clownfish-app-3wxq3.ondigitalocean.app/Home/
3. Click Login → Should redirect to Globus
4. After authentication → Should return and maintain session
5. Navigate to other pages → Session should persist

## Optional: Add Redis for Better Performance

1. In App Platform, add Redis:
   - Add Resource → Add-Ons → Redis
   - This improves session handling and caching

2. The app will automatically use Redis if REDIS_URL is available

## Monitoring

Check Runtime Logs for confirmation:
```
PRODUCTION SETTINGS LOADED
SOCIAL_AUTH_REDIRECT_IS_HTTPS: True
SESSION_ENGINE: django.contrib.sessions.backends.db
DATABASE: PostgreSQL
```

## Why This Fixes the Problem

1. **SOCIAL_AUTH_REDIRECT_IS_HTTPS = True**: OAuth callbacks use HTTPS URLs, setting this to False caused URL mismatches
2. **SESSION_COOKIE_DOMAIN = None**: Allows cookies to work across container restarts
3. **PostgreSQL Database**: Provides persistent session storage across all container instances
4. **SESSION_SAVE_EVERY_REQUEST**: Ensures session state is saved even on read-only requests

## If Issues Persist

1. **Check Logs**: Look for "AuthStateMissing" errors in Runtime Logs
2. **Verify Database**: Ensure PostgreSQL is connected and migrations ran
3. **Clear Sessions**: Run `python manage.py clearsessions` in console
4. **Check OAuth App**: Verify redirect URLs in Globus Auth app settings match your domain

## Alternative Solution (If Above Doesn't Work)

Use sticky sessions by adding to App Spec:
```yaml
services:
  - name: new-uchig
    http_port: 8080
    routes:
      - path: /
    health_check:
      http_path: /
    session_affinity: true  # Add this line
```

This ensures the same user always hits the same container.

---

**The key issue was `SOCIAL_AUTH_REDIRECT_IS_HTTPS = False` - this MUST be True for OAuth to work on HTTPS!**