"""
Production settings fix for OAuth authentication issues on DigitalOcean App Platform
This file contains the corrected settings to resolve session state persistence problems
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# ========================================
# CRITICAL FIX #1: Session Configuration
# ========================================
# Use database-backed sessions for persistence across container restarts
# SQLite won't work properly in a multi-container environment
SESSION_ENGINE = 'django.contrib.sessions.backends.db'  # Changed from cached_db to db only

# Remove domain restriction to avoid cookie issues
SESSION_COOKIE_DOMAIN = None  # Changed from specific domain
SESSION_COOKIE_SECURE = True  # Keep secure for HTTPS
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'  # Changed from None to Lax for better security
SESSION_COOKIE_AGE = 86400  # 24 hours instead of default 2 weeks
SESSION_SAVE_EVERY_REQUEST = True  # Ensure session is saved on every request

# ========================================
# CRITICAL FIX #2: CSRF Configuration
# ========================================
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_SAMESITE = 'Lax'  # Match session cookie setting
CSRF_COOKIE_DOMAIN = None  # Remove domain restriction
CSRF_TRUSTED_ORIGINS = [
    'https://clownfish-app-3wxq3.ondigitalocean.app',
    'https://*.ondigitalocean.app',  # Allow all DO app domains
]

# ========================================
# CRITICAL FIX #3: OAuth Redirect Configuration
# ========================================
# This MUST be True for HTTPS redirects to work properly
SOCIAL_AUTH_REDIRECT_IS_HTTPS = True  # CRITICAL: Changed from False to True

# Ensure proper proxy headers are trusted
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
USE_X_FORWARDED_HOST = True  # Add this to trust forwarded host headers
USE_X_FORWARDED_PORT = True  # Add this to trust forwarded port

# ========================================
# CRITICAL FIX #4: Database Configuration
# ========================================
# For production, you should use PostgreSQL instead of SQLite
# SQLite doesn't work well with multiple containers
DATABASE_URL = os.environ.get('DATABASE_URL')

if DATABASE_URL:
    # Parse DATABASE_URL for PostgreSQL (provided by DigitalOcean)
    import dj_database_url
    DATABASES = {
        'default': dj_database_url.parse(DATABASE_URL, conn_max_age=600)
    }
else:
    # Fallback to SQLite for local development only
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

# ========================================
# Additional OAuth Settings
# ========================================
# Add session refresh on login
SOCIAL_AUTH_SESSION_EXPIRATION = False  # Don't expire session based on OAuth token
SOCIAL_AUTH_LOGIN_REDIRECT_URL = '/Home/'  # Explicit redirect after login
SOCIAL_AUTH_LOGIN_ERROR_URL = '/login-error/'  # Handle login errors
SOCIAL_AUTH_RAISE_EXCEPTIONS = False  # Don't raise exceptions in production

# Pipeline to ensure user session is properly created
SOCIAL_AUTH_PIPELINE = (
    'social_core.pipeline.social_auth.social_details',
    'social_core.pipeline.social_auth.social_uid',
    'social_core.pipeline.social_auth.auth_allowed',
    'social_core.pipeline.social_auth.social_user',
    'social_core.pipeline.user.get_username',
    'social_core.pipeline.user.create_user',
    'social_core.pipeline.social_auth.associate_user',
    'social_core.pipeline.social_auth.load_extra_data',
    'social_core.pipeline.user.user_details',
)

# ========================================
# Security Headers
# ========================================
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# ========================================
# Allowed Hosts
# ========================================
ALLOWED_HOSTS = [
    'clownfish-app-3wxq3.ondigitalocean.app',
    '.ondigitalocean.app',  # Allow all DO app subdomains
    '127.0.0.1',
    'localhost',
]

# ========================================
# Middleware Order (Important!)
# ========================================
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',  # Must be before AuthenticationMiddleware
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'social_django.middleware.SocialAuthExceptionMiddleware',  # Moved before Globus middleware
    'globus_portal_framework.middleware.ExpiredTokenMiddleware',
    'globus_portal_framework.middleware.GlobusAuthExceptionMiddleware',
]

# ========================================
# Cache Configuration (Optional but recommended)
# ========================================
# Use Redis if available for better session handling
REDIS_URL = os.environ.get('REDIS_URL')
if REDIS_URL:
    CACHES = {
        'default': {
            'BACKEND': 'django.core.cache.backends.redis.RedisCache',
            'LOCATION': REDIS_URL,
        }
    }
    # Use cache for sessions if Redis is available
    SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
    SESSION_CACHE_ALIAS = 'default'

print("=" * 50)
print("PRODUCTION SETTINGS LOADED")
print(f"SOCIAL_AUTH_REDIRECT_IS_HTTPS: {SOCIAL_AUTH_REDIRECT_IS_HTTPS}")
print(f"SESSION_ENGINE: {SESSION_ENGINE}")
print(f"SESSION_COOKIE_DOMAIN: {SESSION_COOKIE_DOMAIN}")
print(f"DATABASE: {'PostgreSQL' if DATABASE_URL else 'SQLite (NOT RECOMMENDED)'}")
print("=" * 50)