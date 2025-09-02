"""
Local settings for development
This file overrides settings for local development
"""

# Override HTTPS settings for local development
SOCIAL_AUTH_REDIRECT_IS_HTTPS = False
SECURE_PROXY_SSL_HEADER = None
SESSION_COOKIE_SECURE = False

# Debug mode for local development
DEBUG = True

# Allow localhost
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0']

# Disable CSRF for development (optional, remove if you want CSRF protection)
# CSRF_TRUSTED_ORIGINS = ['http://localhost:8000', 'http://127.0.0.1:8000']

print("Local settings loaded - HTTPS redirect disabled for local development")