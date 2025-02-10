"""
Django settings for myportal project.

Generated by 'django-admin startproject' using Django 3.2.8.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.2/ref/settings/
"""
import logging
import os
from pathlib import Path
from myportal import fields
log = logging.getLogger(__name__)

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Configure the general title for your project
PROJECT_TITLE = 'MADE-PUBLIC Data Portal'
# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.2/howto/deployment/checklist/
# Session and Security Settings
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SOCIAL_AUTH_REDIRECT_IS_HTTPS = True
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_COOKIE_DOMAIN = 'clownfish-app-3wxq3.ondigitalocean.app'

# Login Configuration
LOGIN_URL = '/login/globus'

# SECURITY WARNING: keep all secret keys used in production secret!
# You can generate a secure secret key with `openssl rand -hex 32`
SECRET_KEY = 'a5fbdafa867f7c92f1293664a8e3c12e8c5bc9ffc8f1fdeaacb2d4899a81d1e7'
# Your portal credentials for enabling user login via Globus Auth
SOCIAL_AUTH_GLOBUS_KEY = 'c572c13a-397a-44a5-b744-90a22a8e3b84'
## try to change this line for learning
# learn again
SOCIAL_AUTH_GLOBUS_SECRET = 'g2D/UibXFmzARPKJGnNAHx1TzubrmpDqynXxek7jVbs='

# This is a general Django setting if views need to redirect to login
# https://docs.djangoproject.com/en/3.2/ref/settings/#login-url
LOGIN_URL = '/login/globus'

# This dictates which scopes will be requested on each user login
SOCIAL_AUTH_GLOBUS_SCOPE = [
    'urn:globus:auth:scope:search.api.globus.org:search',
]

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['clownfish-app-3wxq3.ondigitalocean.app', '127.0.0.1', 'localhost']


# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'globus_portal_framework',
    'social_django',
    'sslserver',
    'myportal',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'globus_portal_framework.middleware.ExpiredTokenMiddleware',
    'globus_portal_framework.middleware.GlobusAuthExceptionMiddleware',
    'social_django.middleware.SocialAuthExceptionMiddleware',
]

# Authentication backends setup OAuth2 handling and where user data should be
# stored
AUTHENTICATION_BACKENDS = [
    'globus_portal_framework.auth.GlobusOpenIdConnect',
    'django.contrib.auth.backends.ModelBackend',
]

# CSRF token backend thats authorized
CSRF_TRUSTED_ORIGINS = [
    'https://clownfish-app-3wxq3.ondigitalocean.app',
]

CSRF_COOKIE_SECURE = True

ROOT_URLCONF = 'myportal.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'myportal' / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'globus_portal_framework.context_processors.globals',
            ],
        },
    },
]

WSGI_APPLICATION = 'myportal.wsgi.application'


# Database
# https://docs.djangoproject.com/en/3.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


LOGGING = {
    'version': 1,
    'handlers': {
        'stream': {'level': 'DEBUG', 'class': 'logging.StreamHandler'},
    },
    'loggers': {
        'django': {'handlers': ['stream'], 'level': 'INFO'},
        'globus_portal_framework': {'handlers': ['stream'], 'level': 'INFO'},
        'myportal': {'handlers': ['stream'], 'level': 'INFO'},
    },
}

# Internationalization
# https://docs.djangoproject.com/en/3.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True




# List of search indices managed by the portal
SEARCH_INDEXES = {
    'MADE-PUBLIC Data Transfer': {
        'name': 'MADE-PUBLIC Data Access',
        'uuid': 'f7f1d191-71d7-481a-9d58-cfc2416e3bcb',
        'facets': [
          {
            'name': 'Thrust',
            'field_name': 'Thrust'
          },
          {
            'name': 'Creator',
            'field_name': 'creator'
          },
          {
            'name': 'PI Affliated',
            'field_name': 'PI Affiliated'
          },
          {
            'name': 'Document Format',
            'field_name': 'Document Format'
          },
          {
            'name': 'Data Type',
            'field_name': 'Data Type'
          },
          
          {
                'name': 'Dates',
                'field_name': 'date',
                'type': 'date_histogram',
                'date_interval': 'day',
          },
          {
                'name': 'Months',
                'field_name': 'date',
                'type': 'date_histogram',
                'date_interval': 'month',
          },
          {
                'name': 'Years',
                'field_name': 'date',
                'type': 'date_histogram',
                'date_interval': 'year',
          },
          {
                'name': 'File Sizes',
                'field_name': 'files.length',
                'type': 'numeric_histogram',
                'histogram_range': {'low': 0, 'high': 10000}
          },
          {
            'name': 'Data Tags',
            'field_name': 'Data Tags'
          },
          {
            'name': 'Related Topic',
            'field_name': 'Related Topic'
          },
                ],
        "fields": [
            # Calls a function with your search record as a parameter
            ("title", fields.title),
            ("globus_app_link", fields.globus_app_link),
            ("search_highlights", fields.search_highlights),
            ("https_url", fields.https_url),
            ("dc", fields.dc),
            ("document_format", fields.document_format),
            ("related_topic", fields.related_topic),
            ("data_type", fields.data_type),
            ("abstract_description", fields.abstract_description),
            ("outer_link", fields.outer_link)

        ],
        'facet_modifiers': [
            'globus_portal_framework.modifiers.facets.drop_empty',
        ],

        'boosts': [
            {
                'field_name': 'author',
                'factor': 5
            }
        ],
        'filter_match': 'match-any',
    }
}
# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.2/howto/static-files/


STATIC_URL = '/static/'

# Add these lines
STATICFILES_DIRS = [
    BASE_DIR / "myportal" / "static",
]
STATIC_ROOT = BASE_DIR / "static_root"

# Default primary key field type
# https://docs.djangoproject.com/en/3.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Add these settings to handle media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

try:
    from .local_settings import * # type: ignore
except ImportError:
    expected_path = Path(__file__).resolve().parent / 'local_settings.py'
    log.warning(f'You should create a file for your secrets at {expected_path}')