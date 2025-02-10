from django.shortcuts import render
from django.http import JsonResponse
from .bayesian_optimization import run_BO  # adjust the import based on where you put the function
from io import StringIO
import pandas as pd
from globus_portal_framework.gsearch import (
    get_search_query, get_search_filters,
    process_search_data, get_facets, get_template, get_index
)
from globus_portal_framework.gclients import load_search_client
import json
from django.contrib.auth.decorators import login_required
from globus_portal_framework.gclients import load_transfer_client
from django.shortcuts import redirect
from django.contrib import messages
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from .graph_utils import generate_graph_data
from .forms import BlueprintForm

def Home(request):
    return render(request, 'globus-portal-framework/v2/index-selection.html')

def Team(request):
    return render(request, 'globus-portal-framework/v2/Team.html')

def Research(request):
    return render(request, 'globus-portal-framework/v2/Research.html')

def News(request):
    return render(request, 'globus-portal-framework/v2/News.html')

def Partners(request):
    return render(request, 'globus-portal-framework/v2/Partners.html')

def Repository(request):
    return render(request, 'globus-portal-framework/v2/Repository.html')

def Test_Tagging_System(request):
    return render(request, 'globus-portal-framework/v2/Test Tagging System.html')

def Graph_Visualization(request):
    return render(request, 'globus-portal-framework/v2/Graph_Visualization.html')

def ML_Computing_Toolbox(request):
    return render(request, 'globus-portal-framework/v2/ML_Computing_Toolbox.html')

def Upload_Tutorial(request):
    return render(request, 'globus-portal-framework/v2/Upload_Tutorial.html')

def Cybermanufacturing_Demo(request):
    return render(request, 'globus-portal-framework/v2/Cybermanufacturing_Demo.html')

def Upload_Blueprint_Module(request):
    if request.method == 'POST':
        form = BlueprintForm(request.POST, request.FILES)
        if form.is_valid():
            blueprint = form.save()
            # if request.user.is_authenticated:
            #     blueprint.user = request.user  # Set the user field if authenticated
            blueprint.save()
            messages.success(request, 'Blueprint uploaded successfully.')
            return redirect('Upload_Blueprint_Module')
    else:
        form = BlueprintForm()

    return render(request, 'globus-portal-framework/v2/Upload_Blueprint_Module.html', {'form': form})


def Upload_Data(request):
    if not request.user.is_authenticated:
        # 如果未认证，添加一个消息提示用户登录
        messages.add_message(request, messages.INFO, 'Please log in to upload data.')
        # 使用命名空间构建Globus登录URL，并指定登录后的重定向地址
        login_url = reverse('social:begin', args=['globus']) + '?next=' + reverse('Upload_Data')
        return redirect(login_url)
    # 如果用户已认证，继续正常逻辑
    # ...（你的上传数据逻辑）...
    return render(request, 'globus-portal-framework/v2/Upload_Data.html')


# Add more views here...


def run_bayesian_optimization(request):
    if request.method == 'POST':
        
        feature_start = int(request.POST.get('featureStart'))
        feature_end = int(request.POST.get('featureEnd'))
        output_index = int(request.POST.get('outputIndex'))
        n_calls = int(request.POST.get('nCalls'))
        n_random_starts = int(request.POST.get('nRandomStarts'))
        acq_func = request.POST.get('acqFunc')
        seed = int(request.POST.get('seed'))
        print(feature_start,feature_end,output_index,n_calls,n_random_starts,acq_func,seed)
        # Reading uploaded file
        csv_file = request.FILES.get('csvFile')
        if csv_file is not None:
            csv_data = csv_file.read().decode('UTF-8')
            io_string = StringIO(csv_data)
            df = pd.read_csv(io_string)
            # Do whatever you want to do with df here
        print(request.POST)
        # Parsing bounds
        raw_bounds_str = request.POST.get('bounds')  # Get the JSON-formatted string
        print("Raw Bounds String:", raw_bounds_str)  # for debugging
        raw_bounds = json.loads(raw_bounds_str)  # Deserialize JSON to Python object
        print("Deserialized Bounds:", raw_bounds)  # for debugging

        bounds = [tuple(map(float, bound.split(','))) for bound in raw_bounds]
        print("Converted Bounds:", bounds)  # additional debug log

        # Run Bayesian Optimization
        best_params, best_value = run_BO(seed=seed, 
                                         csv_data=df,  # Passing DataFrame
                                         feature_indexes=(feature_start, feature_end), 
                                         output_index=output_index, 
                                         n_calls=n_calls, 
                                         n_random_starts=n_random_starts, 
                                         acq_func=acq_func, 
                                         bounds=bounds)

        return JsonResponse({
            'best_parameters': best_params,
            'best_objective': best_value
        })
    else:
        return JsonResponse({'error': 'Only POST method is allowed'})
INDEX_MAP = {
    'MADE-PUBLIC-Data-Transfer': 'MADE-PUBLIC Data Transfer',
    # Add other slugs and their corresponding original names here
}
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO 8601 string
        return super().default(obj)
    
def my_advanced_search(request, index):
    # Attempt to get index data, which must be defined in settings.SEARCH_INDEXES
    # index = INDEX_MAP.get(index, index)  # Convert slug back to original name if necessary
    print("the index is:",index)
    index_data = get_index(index)

    search_cli = load_search_client(request.user)
    query = get_search_query(request)
    filters = get_search_filters(request)
    data = {'q': query,
            'filters': filters}
    result = search_cli.post_search(index_data['uuid'], data)
    search_data = {
        'search_results': process_search_data(index_data.get('fields', []),
                                              result.data['gmeta']),
        'facets': get_facets(result, index_data.get('facets', []),
                             filters, index_data.get('filter_match')),
    }

    # Save search results to a file
    with open('search_results.txt', 'w') as file:
        file.write(json.dumps(search_data['search_results'], indent=4, cls=DateTimeEncoder))

    context = {'search': search_data}
    return render(request, get_template(index, 'search.html'), context)

@csrf_exempt
def process_graph_data(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        graph_data = generate_graph_data(data)
        return JsonResponse(graph_data)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def graph_visualization(request):
    # The graph data is no longer passed through the URL
    return render(request, 'globus-portal-framework/v2/Graph_Visualization.html')

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import json
import pickle
import base64
import globus_sdk
from globus_sdk.scopes import AuthScopes
from globus_compute_sdk import Client, Executor
from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk.sdk.login_manager import AuthorizerLoginManager
from globus_compute_sdk.sdk.login_manager.manager import ComputeScopeBuilder
import time
from datetime import date

@csrf_exempt
def test_globus_compute(request):
    if request.method == 'POST':
        tutorial_endpoint = '7db36379-95ad-4892-aee2-43f3a825d275'  # Public tutorial endpoint
        
        # Try to get Globus Auth token data from the session
        globus_data_raw = request.session.get("GLOBUS_DATA")
        print(globus_data_raw)
        if globus_data_raw:
            tokens = pickle.loads(base64.b64decode(globus_data_raw))['tokens']
            
            ComputeScopes = ComputeScopeBuilder()
            
            # Create Authorizers from the Compute and Auth tokens
            compute_auth = globus_sdk.AccessTokenAuthorizer(tokens[ComputeScopes.resource_server]['access_token'])
            openid_auth = globus_sdk.AccessTokenAuthorizer(tokens['auth.globus.org']['access_token'])
            
            # auth_client = globus_sdk.AuthClient(authorizer=openid_auth)
            # user_info = auth_client.oauth2_userinfo()
            # print(f"Authenticated User: {user_info}")

            # Create a Compute Client from these authorizers
            compute_login_manager = AuthorizerLoginManager(
                authorizers={ComputeScopes.resource_server: compute_auth,
                             AuthScopes.resource_server: openid_auth}
            )
            compute_login_manager.ensure_logged_in()
            gc = Client(login_manager=compute_login_manager, code_serialization_strategy=CombinedCode())
            gce = Executor(endpoint_id=tutorial_endpoint, client=gc)
        else:
            # Create an Executor and initiate an auth flow if tokens are not available
            gc = Client(code_serialization_strategy=CombinedCode())
            gce = Executor(endpoint_id=tutorial_endpoint, client=gc)
        
        results = {}


        # Test 1: Hello World
        def hello_world():
            import getpass
            import os
            return f"Hello World from {getpass.getuser()} on {os.uname().nodename}"
        
        future = gce.submit(hello_world)
        results['hello_world'] = future.result()

        # Test 2: Division by zero (exception handling)
        def division_by_zero():
            return 42 / 0
        
        future = gce.submit(division_by_zero)
        try:
            future.result()
        except Exception as exc:
            results['division_by_zero'] = str(exc)

        # Test 3: Simple addition
        def get_sum(a, b):
            return a + b
        
        future = gce.submit(get_sum, 40, 2)
        results['addition'] = future.result()

        # Test 4: Get current date
        def get_date():
            from datetime import date  # Import date within the function
            return str(date.today())

        future = gce.submit(get_date)
        results['current_date'] = future.result()

        # Test 5: Echo command
        def echo(name):
            import os
            return os.popen(f"echo Hello {name} from $HOSTNAME at $PWD").read().strip()
        
        future = gce.submit(echo, "World")
        results['echo'] = future.result()

        # Test 6: ollama test
        def run_ollama(prompt):
            import subprocess
            import time

            start_time = time.time()
            try:
                #630 second timeout for the subprocess
                result = subprocess.run(['ollama', 'run', 'llama3.1', prompt], 
                                        capture_output=True, text=True, timeout=60)
                output = result.stdout.strip()
            except subprocess.TimeoutExpired:
                output = "Ollama command timed out after 60 seconds"
            except subprocess.CalledProcessError as e:
                output = f"Ollama command failed: {e}"
            except Exception as e:
                output = f"An error occurred: {str(e)}"
            
            end_time = time.time()
            execution_time = end_time - start_time

            return {
                'output': output,
                'execution_time': execution_time
            }

        prompt = "Please describe very briefly, based on your knowledge what is FET sensor."
        future = gce.submit(run_ollama, prompt)
        results['ollama'] = future.result()



        # Test 7: Pi estimation
        def pi(num_points):
            from random import random
            inside = 0   
            
            for i in range(num_points):
                x, y = random(), random()  # Drop a point randomly within the box.
                if x**2 + y**2 < 1:        # Count points within the circle.
                    inside += 1  
            return (inside*4 / num_points)

        estimates = []
        for _ in range(3):
            estimates.append(gce.submit(pi, 10**5))
        
        pi_results = [future.result() for future in estimates]
        results['pi_estimation'] = {
            'estimates': pi_results,
            'average': sum(pi_results) / len(pi_results)
        }

        return JsonResponse({
            "message": "Globus Compute tests completed",
            "results": results
        })
    
    return JsonResponse({"error": "Invalid request method"}, status=400)

from django.shortcuts import redirect
from django.urls import reverse
from django.contrib import messages
import base64, pickle, json, os
from django.views.decorators.csrf import csrf_exempt
from globus_compute_sdk import Client, Executor
from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk.sdk.login_manager import AuthorizerLoginManager
from globus_compute_sdk.sdk.login_manager.manager import ComputeScopeBuilder
from globus_sdk.scopes import AuthScopes  # ensure AuthScopes is imported
from django.http import JsonResponse
from django.conf import settings

@csrf_exempt
def run_ollama_interactive(request):
    if request.method == 'POST':
        # Check for Globus token data in the session.
        globus_data_raw = request.session.get("GLOBUS_DATA")
        if not globus_data_raw:
            # Redirect to a web-based Globus login view.
            # (Make sure you have a view configured to handle Globus login.)
            login_url = reverse('social:begin', args=['globus']) + '?next=' + request.get_full_path()
            messages.error(request, "Globus authentication required. Please log in.")
            return redirect(login_url)

        # Continue with your existing logic.
        data = json.loads(request.body)
        prompt = data.get('prompt', '')
        conversation_history = data.get('conversation_history', [])
        is_initial_message = data.get('is_initial_message', False)

        if is_initial_message:
            file_path = os.path.join(settings.BASE_DIR, 'myportal', 'static', 'json', 'background_knowledge.txt')
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    background_knowledge = file.read()
                # Add background knowledge to conversation history
                conversation_history = [{'role': 'System', 'content': background_knowledge}]
                prompt = "Acknowledge that you've received the background knowledge."
            except FileNotFoundError:
                print(f"Background knowledge file not found at {file_path}")
                conversation_history = []
                prompt = "Unable to load background knowledge. Please proceed with the conversation."
            except UnicodeDecodeError:
                print(f"Error decoding the background knowledge file. Please ensure it's saved in UTF-8 encoding.")
                conversation_history = []
                prompt = "Unable to load background knowledge due to encoding issues. Please proceed with the conversation."

        if not prompt:
            return JsonResponse({'error': 'No prompt provided'}, status=400)

        # Append the prompt if not already the last Human message.
        if not (conversation_history and conversation_history[-1]['role'] == 'Human' and conversation_history[-1]['content'] == prompt):
            conversation_history.append({'role': 'Human', 'content': prompt})

        # Load token data from session.
        tokens = pickle.loads(base64.b64decode(globus_data_raw))['tokens']
        ComputeScopes = ComputeScopeBuilder()
        compute_auth = globus_sdk.AccessTokenAuthorizer(tokens[ComputeScopes.resource_server]['access_token'])
        openid_auth = globus_sdk.AccessTokenAuthorizer(tokens['auth.globus.org']['access_token'])
        compute_login_manager = AuthorizerLoginManager(
            authorizers={
                ComputeScopes.resource_server: compute_auth,
                AuthScopes.resource_server: openid_auth
            }
        )
        # Ensure the user is logged in via Globus (this should be a no-op if tokens are valid).
        compute_login_manager.ensure_logged_in()
        gc = Client(login_manager=compute_login_manager, code_serialization_strategy=CombinedCode())
        tutorial_endpoint = '7db36379-95ad-4892-aee2-43f3a825d275'
        gce = Executor(endpoint_id=tutorial_endpoint, client=gc)

        # Define the function to run ollama.
        def run_ollama(prompt, conversation_history):
            import subprocess, time
            start_time = time.time()
            # Prepare the full conversation context
            full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
            full_prompt += "\nAI:"
            try:
                result = subprocess.run(
                    ['ollama', 'run', 'llama3.1', full_prompt],
                    capture_output=True, text=True, timeout=60
                )
                output = result.stdout.strip()
            except subprocess.TimeoutExpired:
                output = "Ollama command timed out after 60 seconds"
            except subprocess.CalledProcessError as e:
                output = f"Ollama command failed: {e}"
            except Exception as e:
                output = f"An error occurred: {str(e)}"
            execution_time = time.time() - start_time
            return {
                'output': output,
                'execution_time': execution_time
            }

        future = gce.submit(run_ollama, prompt, conversation_history)
        result = future.result()
        # Append AI response to conversation history.
        conversation_history.append({'role': 'AI', 'content': result['output']})

        return JsonResponse({
            'output': result['output'],
            'execution_time': result['execution_time'],
            'conversation_history': conversation_history
        })

    return JsonResponse({'error': 'Only POST method is allowed'}, status=400)
    
import re

def extract_metadata_summary(json_data):
    gmeta_data = json_data['ingest_data']['gmeta']
    num_nodes = len(gmeta_data)
    
    summary = f"These are the metadata files of {num_nodes} data file target(s) in the large project repository database:\n\n"
    
    for item in gmeta_data:
        metadata = item['all'][0]
        
        # Extract directory information from https_url
        url_path = item['https_url']
        dir_info = re.search(r'MADE_PUBLIC_REPO(.+)$', url_path)
        dir_path = dir_info.group(1) if dir_info else "N/A"
        
        summary += f"Title: {metadata['Title']}\n"
        summary += f"Creator: {metadata['creator']}\n"
        summary += f"Data Type: {', '.join(metadata['Data Type'])}\n"
        summary += f"Related Topic: {', '.join(metadata['Related Topic'])}\n"
        summary += f"Thrust: {metadata['Thrust']}\n"
        summary += f"Data Tags: {', '.join(metadata['Data Tags'])}\n"
        summary += f"Date: {metadata['date']}\n"
        summary += f"Directory: {dir_path}\n\n"
    
    summary += """Could you provide a high-level summary of these items based on their metadata, e.g. their topic, what type of data they are, what commonalities are there, which thrust they belong (or some belong to xx, some belong to xx), what is the best way to utilize these data for what purpose etc."""

    return summary

@csrf_exempt
def extract_metadata_summary_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        summary = extract_metadata_summary(data)
        return JsonResponse({'summary': summary})
    return JsonResponse({'error': 'Invalid request method'}, status=400)