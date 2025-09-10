"""
Template for local_settings.py
Copy this file to local_settings.py and add your API keys
DO NOT COMMIT local_settings.py TO GIT
"""

# Override HTTPS settings for local development
SOCIAL_AUTH_REDIRECT_IS_HTTPS = False
SECURE_PROXY_SSL_HEADER = None
SESSION_COOKIE_SECURE = False

# Debug mode for local development
DEBUG = True

# Allow localhost
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0']

# API Keys - ADD YOUR KEYS HERE
XAI_API_KEY = 'your-xai-api-key-here'

print("Local settings loaded - HTTPS redirect disabled for local development")