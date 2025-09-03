"""
URL configuration for myportal project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import views 
from myportal import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('Home/', views.Home, name='Home'),
    path('admin/', admin.site.urls),
    path('Team/', views.Team, name='Team'),
    path('Research/', views.Research, name='Research'),
    path('News/', views.News, name='News'),
    path('Partners/', views.Partners, name='Partners'),
    path('Repository/', views.Repository, name='Repository'),
    path('Graph_Visualization/', views.Graph_Visualization, name='Graph_Visualization'),
    path('ML_Computing_Toolbox/', views.ML_Computing_Toolbox, name='ML_Computing_Toolbox'),
    path('Upload_Data/', views.Upload_Data, name='Upload_Data'),
    path('Upload_Tutorial/', views.Upload_Tutorial, name='Upload_Tutorial'),
    path('Test_Tagging_System/', views.Test_Tagging_System, name='Test_Tagging_System'),
    path('Cybermanufacturing_Demo/', views.Cybermanufacturing_Demo, name='Cybermanufacturing_Demo'),
    path('Upload_Blueprint_Module/', views.Upload_Blueprint_Module, name='Upload_Blueprint_Module'),

    path('run_bayesian_optimization/', views.run_bayesian_optimization, name='run_bayesian_optimization'),
    # ... other paths
    path('search/<str:index>/', views.my_advanced_search, name='adv_search'),
    path('', include('globus_portal_framework.urls')),
    path('', include('social_django.urls', namespace='social')),
    path('process_graph_data/', views.process_graph_data, name='process_graph_data'),
    path('graph_visualization/', views.graph_visualization, name='graph_visualization'),
    path('test-globus-compute/', views.test_globus_compute, name='test_globus_compute'),
    path('run_ollama_interactive/', views.run_ollama_interactive, name='run_ollama_interactive'),
    path('extract_metadata_summary/', views.extract_metadata_summary_view, name='extract_metadata_summary'),
    path('auto_ingest_metadata/', views.auto_ingest_metadata, name='auto_ingest_metadata'),
    path('auto_transfer_files/', views.auto_transfer_files, name='auto_transfer_files'),
    path('upload_files_for_tagging/', views.upload_files_for_tagging, name='upload_files_for_tagging'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)