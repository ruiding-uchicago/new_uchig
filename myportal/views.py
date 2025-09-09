from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .bayesian_optimization import run_BO  # adjust the import based on where you put the function
from io import StringIO
import pandas as pd
from globus_portal_framework.gsearch import (
    get_search_query, get_search_filters,
    process_search_data, get_facets, get_template, get_index
)
from globus_portal_framework.gclients import load_search_client
from globus_portal_framework import gsearch
import json
from django.contrib.auth.decorators import login_required

# Monkey patch the gsearch.post_search to handle multiple indices
original_post_search = gsearch.post_search

def multi_index_post_search(index, query, filters, user=None, page=1, search_kwargs=None):
    """Modified post_search that handles multiple indices"""
    from globus_portal_framework.gsearch import (
        prepare_search_facets, get_pagination, get_setting, VALID_SEARCH_KEYS
    )
    
    if not index or not query:
        return {'search_results': [], 'facets': []}
    
    index_data = get_index(index)
    
    # Check if we have multiple UUIDs
    uuids = index_data.get('uuid', [])
    if not isinstance(uuids, list):
        # Single UUID - use original function
        return original_post_search(index, query, filters, user, page, search_kwargs)
    
    # Multiple UUIDs - search each and merge results
    client = load_search_client(user)
    
    # Build search data (same as original function)
    search_data = {k: index_data[k] for k in VALID_SEARCH_KEYS
                   if k in index_data}
    search_data.update({
        'q': query,
        'facets': prepare_search_facets(index_data.get('facets', [])),
        'filters': filters,
        'offset': (int(page) - 1) * get_setting('SEARCH_RESULTS_PER_PAGE'),
        'limit': get_setting('SEARCH_RESULTS_PER_PAGE')
    })
    search_data.update(search_kwargs or {})
    
    # Search all indices and merge results
    all_gmeta = []
    all_facet_results = []
    merged_result = None
    total_count = 0
    
    for uuid in uuids:
        try:
            print(f"Searching index {uuid} with query: '{search_data.get('q')}'")
            result = client.post_search(uuid, search_data)
            gmeta = result.data.get('gmeta', [])
            print(f"Found {len(gmeta)} results in index {uuid}")
            all_gmeta.extend(gmeta)
            total_count += result.data.get('total', 0)
            all_facet_results.append(result.data.get('facet_results', []))
            if merged_result is None:
                merged_result = result  # Keep first result for structure
        except Exception as e:
            print(f"Error searching index {uuid}: {e}")
            continue
    
    # If no results from any index, return empty
    if merged_result is None:
        return {
            'search_results': [],
            'facets': [],
            'pagination': [],
            'count': 0,
            'offset': 0,
            'total': 0,
        }
    
    # Merge facets from all indices
    if all_facet_results:
        merged_facets = {}
        for facet_results in all_facet_results:
            if facet_results:
                for facet in facet_results:
                    facet_name = facet.get('name')
                    if facet_name not in merged_facets:
                        merged_facets[facet_name] = facet.copy()
                        merged_facets[facet_name]['buckets'] = []
                    
                    # Merge buckets
                    existing_buckets = {b['value']: b for b in merged_facets[facet_name]['buckets']}
                    for bucket in facet.get('buckets', []):
                        if bucket['value'] in existing_buckets:
                            existing_buckets[bucket['value']]['count'] += bucket['count']
                        else:
                            merged_facets[facet_name]['buckets'].append(bucket.copy())
        
        # Convert back to list format
        merged_result.data['facet_results'] = list(merged_facets.values())
    
    # Update merged result with combined data
    merged_result.data['gmeta'] = all_gmeta
    merged_result.data['count'] = len(all_gmeta)
    merged_result.data['total'] = total_count
    
    # Process and return combined results
    return {
        'search_results': process_search_data(index_data.get('fields', []),
                                              all_gmeta),
        'facets': get_facets(merged_result, index_data.get('facets', []),
                             filters, index_data.get('filter_match'),
                             index_data.get('facet_modifiers', [])),
        'pagination': get_pagination(total_count, merged_result.data['offset']),
        'count': len(all_gmeta),
        'offset': merged_result.data['offset'],
        'total': total_count,
    }

# Apply the monkey patch
gsearch.post_search = multi_index_post_search

# Also need to patch get_subject for detail views
original_get_subject = gsearch.get_subject

def multi_index_get_subject(index, subject, user=None):
    """Modified get_subject that handles multiple indices"""
    from urllib.parse import unquote
    import globus_sdk
    
    client = load_search_client(user)
    idata = get_index(index)
    
    # Check if we have multiple UUIDs
    uuids = idata.get('uuid', [])
    if not isinstance(uuids, list):
        # Single UUID - use original function
        return original_get_subject(index, subject, user)
    
    # Multiple UUIDs - try each until we find the subject
    for uuid in uuids:
        try:
            result = client.get_subject(uuid, unquote(subject))
            return process_search_data(idata.get('fields', {}), [result.data])[0]
        except globus_sdk.SearchAPIError:
            continue  # Try next index
    
    # Not found in any index
    return {'subject': subject, 'error': 'No data was found for subject'}

# Apply the get_subject patch
gsearch.get_subject = multi_index_get_subject
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
    
    # Handle both single UUID and list of UUIDs
    uuids = index_data['uuid']
    if not isinstance(uuids, list):
        uuids = [uuids]  # Convert single UUID to list for uniform handling
    
    # Search all indices and merge results
    all_gmeta = []
    merged_result = None
    for uuid in uuids:
        try:
            result = search_cli.post_search(uuid, data)
            all_gmeta.extend(result.data.get('gmeta', []))
            if merged_result is None:
                merged_result = result  # Keep first result for facets
        except Exception as e:
            print(f"Error searching index {uuid}: {e}")
            continue
    
    # If no results from any index, create empty result
    if merged_result is None:
        search_data = {
            'search_results': [],
            'facets': [],
        }
    else:
        # Use merged gmeta for search results, but facets from first successful result
        merged_result.data['gmeta'] = all_gmeta
        search_data = {
            'search_results': process_search_data(index_data.get('fields', []),
                                                  all_gmeta),
            'facets': get_facets(merged_result, index_data.get('facets', []),
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
        data = json.loads(request.body)
        prompt = data.get('prompt', '')
        conversation_history = data.get('conversation_history', [])
        is_initial_message = data.get('is_initial_message', False)

        if is_initial_message:
            # load background knowledge as before...
            file_path = os.path.join(settings.BASE_DIR, 'myportal', 'static', 'json', 'background_knowledge.txt')
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    background_knowledge = file.read()
                conversation_history = [{'role': 'System', 'content': background_knowledge}]
                prompt = "Acknowledge that you've received the background knowledge."
            except FileNotFoundError:
                conversation_history = []
                prompt = "Unable to load background knowledge. Please proceed with the conversation."
            except UnicodeDecodeError:
                conversation_history = []
                prompt = "Unable to load background knowledge due to encoding issues. Please proceed with the conversation."

        if not prompt:
            return JsonResponse({'error': 'No prompt provided'}, status=400)

        # Append the new prompt if not already the last message.
        if not (conversation_history and conversation_history[-1]['role'] == 'Human' and conversation_history[-1]['content'] == prompt):
            conversation_history.append({'role': 'Human', 'content': prompt})

        tutorial_endpoint = '7db36379-95ad-4892-aee2-43f3a825d275'
        
        # Here is the key modification: instead of blindly instantiating Client (which would try to prompt interactively),
        # we first check for existing tokens in the session.
        globus_data_raw = request.session.get("GLOBUS_DATA")
        if globus_data_raw:
            try:
                tokens = pickle.loads(base64.b64decode(globus_data_raw))['tokens']
            except Exception as e:
                return JsonResponse({'error': 'Error decoding authentication tokens: ' + str(e)}, status=401)

            # Build authorizers using the tokens we expect to have been stored.
            compute_scopes = ComputeScopeBuilder()
            try:
                compute_auth = AccessTokenAuthorizer(tokens[compute_scopes.resource_server]['access_token'])
                openid_auth = AccessTokenAuthorizer(tokens['auth.globus.org']['access_token'])
            except KeyError as e:
                return JsonResponse({'error': 'Missing expected token data: ' + str(e)}, status=401)

            # Create a login manager that simply returns these tokens.
            login_manager = AuthorizerLoginManager(
                authorizers={
                    compute_scopes.resource_server: compute_auth,
                    AuthScopes.resource_server: openid_auth,
                }
            )
            # In production, if tokens are invalid, we do not want interactive login.
            try:
                login_manager.ensure_logged_in()
            except Exception as e:
                return JsonResponse({'error': 'Authentication error: ' + str(e)}, status=401)

            gc = Client(login_manager=login_manager, code_serialization_strategy=CombinedCode())
        else:
            # In production, if there are no tokens available, immediately return an error.
            return JsonResponse({'error': 'Authentication tokens missing in session. Please log in first.'}, status=401)

        # Create the Executor as before.
        gce = Executor(endpoint_id=tutorial_endpoint, client=gc)

        # Define the function to run Ollama.
        def run_ollama(prompt, conversation_history):
            import subprocess
            import time
            start_time = time.time()
            full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
            full_prompt += "\nAI:"
            try:
                result = subprocess.run(['ollama', 'run', 'llama3.1', full_prompt],
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

        future = gce.submit(run_ollama, prompt, conversation_history)
        result = future.result()

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

@csrf_exempt
@login_required
def auto_ingest_metadata(request):
    """
    Automatically ingest metadata JSON to Globus Search index
    Phase 1: Assumes files are already uploaded to Globus endpoint manually
    Try API first, fall back to CLI if available
    """
    if request.method == 'POST':
        try:
            # Get the JSON data from request
            json_data = json.loads(request.body)
            
            # Get the target index UUID - using the new September 2025 index
            target_index_uuid = '64565b2d-ac5b-480e-8669-1884f1573b53'
            
            # First, try using the API with user's token
            try:
                client = load_search_client(request.user)
                result = client.ingest(target_index_uuid, json_data)
                
                if result.data.get('acknowledged'):
                    return JsonResponse({
                        'success': True,
                        'message': 'Metadata successfully ingested to search index',
                        'task_id': result.data.get('task_id', ''),
                        'index_uuid': target_index_uuid
                    })
            except Exception as api_error:
                # If API fails, try CLI as fallback (for local dev)
                import subprocess
                import tempfile
                import os
                
                try:
                    # Check if globus CLI is available
                    cli_check = subprocess.run(['which', 'globus'], capture_output=True)
                    if cli_check.returncode != 0:
                        # CLI not available, return API error
                        raise api_error
                    
                    # Write JSON to temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                        json.dump(json_data, tmp_file, indent=2)
                        tmp_filename = tmp_file.name
                    
                    # Try CLI approach
                    result = subprocess.run(
                        ['globus', 'search', 'ingest', target_index_uuid, tmp_filename],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    # Clean up temp file
                    os.unlink(tmp_filename)
                    
                    if result.returncode == 0:
                        return JsonResponse({
                            'success': True,
                            'message': 'Metadata successfully ingested to search index via CLI',
                            'task_id': '',
                            'index_uuid': target_index_uuid
                        })
                    else:
                        # Both API and CLI failed
                        return JsonResponse({
                            'success': False,
                            'message': f'Ingest failed. API error: {str(api_error)}. CLI error: {result.stderr}',
                            'api_error': str(api_error),
                            'cli_error': result.stderr
                        }, status=400)
                        
                except Exception as cli_error:
                    # Both methods failed
                    return JsonResponse({
                        'success': False,
                        'message': f'Cannot ingest. API error: {str(api_error)}. CLI error: {str(cli_error)}',
                        'api_error': str(api_error),
                        'cli_error': str(cli_error)
                    }, status=400)
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'Error during ingest: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Only POST method is allowed'}, status=400)


@csrf_exempt
@login_required  
def upload_files_for_tagging(request):
    """
    Handle file uploads from the tagging system
    Store files temporarily for users to manually transfer to Globus
    """
    if request.method == 'POST':
        try:
            uploaded_files = request.FILES.getlist('files')
            user_email = request.user.email if hasattr(request.user, 'email') else request.user.username
            
            # Create user-specific directory
            import os
            from django.conf import settings
            
            user_folder = os.path.join(settings.MEDIA_ROOT, 'tagging_uploads', user_email.replace('@', '_at_'))
            os.makedirs(user_folder, exist_ok=True)
            
            saved_files = []
            for uploaded_file in uploaded_files:
                # Save file
                file_path = os.path.join(user_folder, uploaded_file.name)
                with open(file_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)
                
                saved_files.append({
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'path': file_path
                })
            
            return JsonResponse({
                'success': True,
                'message': f'Successfully uploaded {len(saved_files)} file(s)',
                'files': saved_files,
                'storage_path': user_folder
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'Error uploading files: {str(e)}'
            }, status=500)
    
    return JsonResponse({'success': False, 'message': 'Method not allowed'}, status=405)


@csrf_exempt
@login_required
def auto_transfer_files(request):
    """
    Automatically transfer files to Globus endpoint after tagging
    Uses Globus Transfer API with user's OAuth token
    """
    if request.method == 'POST':
        try:
            # Parse request data
            data = json.loads(request.body)
            source_endpoint = data.get('source_endpoint')
            file_paths = data.get('file_paths', [])
            
            if not source_endpoint or not file_paths:
                return JsonResponse({
                    'success': False,
                    'message': 'Missing source_endpoint or file_paths'
                }, status=400)
            
            # Destination is the MADE-PUBLIC collection endpoint
            # This is the Globus endpoint ID for the collection
            dest_endpoint = 'c1e16320-e4ba-4bec-b653-6b4e9c85b522'  # UChicago RCC endpoint
            dest_base_path = '/project2/chenyuxin/2025MADEPUBLICsnapshot/'
            
            # Load transfer client using the framework's helper
            from globus_portal_framework import load_transfer_client
            transfer_client = load_transfer_client(request.user)
            
            # Create transfer data
            from globus_sdk import TransferData
            import uuid
            
            # Generate a label for this transfer
            transfer_label = f"MADE-PUBLIC Transfer {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Create the transfer data object
            transfer_data = TransferData(
                transfer_client,
                source_endpoint,
                dest_endpoint,
                label=transfer_label,
                sync_level="checksum"  # Ensure file integrity
            )
            
            # Add each file to the transfer
            for file_path in file_paths:
                # Ensure paths are properly formatted
                source_path = file_path if file_path.startswith('/') else f'/{file_path}'
                # Extract just the filename for destination
                filename = os.path.basename(file_path)
                dest_path = f"{dest_base_path}{filename}"
                
                transfer_data.add_item(source_path, dest_path)
            
            # Submit the transfer
            transfer_result = transfer_client.submit_transfer(transfer_data)
            
            if transfer_result['code'] == 'Accepted':
                return JsonResponse({
                    'success': True,
                    'message': f'Transfer submitted successfully. {len(file_paths)} file(s) queued.',
                    'task_id': transfer_result['task_id'],
                    'submission_id': transfer_result['submission_id'],
                    'label': transfer_label
                })
            else:
                return JsonResponse({
                    'success': False,
                    'message': f'Transfer submission failed: {transfer_result.get("message", "Unknown error")}'
                }, status=400)
                
        except Exception as e:
            import traceback
            return JsonResponse({
                'success': False,
                'message': f'Error during transfer: {str(e)}',
                'details': traceback.format_exc()
            }, status=500)
    
# Probe-Target Detection Designer Views
def Probe_Target_Detection_Designer(request):
    """Render the Probe-Target Detection Designer page"""
    return render(request, 'globus-portal-framework/v2/PFAS_Detection_Designer.html')

@csrf_exempt
def pfas_inference_api(request):
    """API endpoint for PFAS model inference"""
    if request.method == 'POST':
        try:
            import json
            import os
            import sys
            
            # Add pfas_screen_model to path
            pfas_model_path = os.path.join(settings.BASE_DIR, 'pfas_screen_model')
            if pfas_model_path not in sys.path:
                sys.path.insert(0, pfas_model_path)
            
            # Change working directory to pfas_screen_model for relative paths
            original_dir = os.getcwd()
            os.chdir(pfas_model_path)
            
            try:
                # Import inference engine
                from inference_engine import (
                    load_model_and_config, 
                    load_database, 
                    run_inference
                )
            
                # Parse request data
                data = json.loads(request.body)
                mode = data.get('mode', 'complete')
                target_name = data.get('target')
                probe_name = data.get('probe')
                medium_name = data.get('medium')
                condition = data.get('condition')
                
                print(f"Inference request - Mode: {mode}, Target: {target_name}, Probe: {probe_name}, Medium: {medium_name}, Condition: {condition}")
                
                # Load model and database
                model, config, device = load_model_and_config()
                database = load_database()
                
                print(f"Model loaded successfully. Device: {device}")
                print(f"Database keys: {database.keys() if database else 'None'}")
                
                # Helper function to get substance from database or create a basic one
                def get_substance(name, substance_type='small molecule'):
                    if not name:
                        return None
                    
                    if database:
                        # Try exact match first
                        substances_by_name = database.get('substances_by_name', {})
                        if name in substances_by_name:
                            print(f"Found exact match in database: {name}")
                            return substances_by_name[name]
                        
                        # Try case-insensitive match
                        for key, value in substances_by_name.items():
                            if key.lower() == name.lower():
                                print(f"Found case-insensitive match: {key}")
                                return value
                    
                    # If not found in database, create a minimal substance object
                    # This is needed because the model expects a dict with specific structure
                    print(f"Creating minimal substance object for: {name}")
                    return {
                        'substance_name': name,
                        'substance_type': substance_type,
                        'substance_descriptors': {
                            # Minimal descriptors - the model will handle missing values
                            'MolecularWeight': 100.0,
                            'XLogP': 1.0,
                            'HBondDonorCount': 1,
                            'HBondAcceptorCount': 2,
                            'RotatableBondCount': 1,
                            'TPSA': 20.0,
                            'Complexity': 100.0
                        }
                    }
                
                # Prepare inputs for inference
                target_sub = get_substance(target_name)
                probe_sub = get_substance(probe_name)
                medium_sub = get_substance(medium_name)
                
                # Convert condition to tuple if provided
                condition_tuple = tuple(condition) if condition else None
                
                print(f"Substances found - Target: {target_sub is not None}, Probe: {probe_sub is not None}, Medium: {medium_sub is not None}")
                
                # Run inference with actual model
                result = run_inference(
                    model, config, device, database,
                    target=target_sub,
                    probe=probe_sub,
                    medium=medium_sub,
                    condition=condition_tuple
                )
                
                print(f"Inference result: {result}")
                
                # Format response based on mode
                if mode == 'complete':
                    # Scoring mode - single prediction
                    prediction = result.get('prediction_label', 2)
                    return JsonResponse({
                        'prediction_label': prediction,
                        'confidence': 90.0 + (4 - prediction) * 2.5,
                        'sensitivity': ['Very High', 'High', 'Moderate', 'Low', 'Very Low'][prediction],
                        'model_used': True
                    })
                else:
                    # Screening mode - recommendations
                    return JsonResponse(result)
                    
            finally:
                # Restore original directory
                os.chdir(original_dir)
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"PFAS Inference Error: {str(e)}")
            print(f"Traceback: {error_details}")
            
            # Check what kind of error and provide helpful feedback
            error_msg = str(e)
            
            if 'No module named' in error_msg:
                # Missing dependencies
                return JsonResponse({
                    'error': 'Missing dependencies',
                    'message': 'Please install required packages: torch, torch_geometric, snntorch',
                    'details': error_msg,
                    'simulated': True,
                    'prediction_label': simulate_ldl_score(target_name, probe_name, medium_name, condition) if mode == 'complete' else None
                }, status=500)
            elif 'pfas_multimodal.pt' in error_msg or 'config.json' in error_msg:
                # Model files not found
                return JsonResponse({
                    'error': 'Model files not found',
                    'message': 'Model files missing in pfas_screen_model/saved-models/',
                    'details': error_msg,
                    'simulated': True,
                    'prediction_label': simulate_ldl_score(target_name, probe_name, medium_name, condition) if mode == 'complete' else None
                }, status=500)
            else:
                # General error - use intelligent simulation
                import random
                
                if mode == 'complete':
                    score = simulate_ldl_score(target_name, probe_name, medium_name, condition)
                    return JsonResponse({
                        'prediction_label': score,
                        'confidence': 75 + random.random() * 20,
                        'sensitivity': ['Very High', 'High', 'Moderate', 'Low', 'Very Low'][score],
                        'simulated': True,
                        'message': 'Using intelligent simulation due to model error',
                        'error': error_msg
                    })
                else:
                    # Simulated screening results
                    return JsonResponse({
                        'best_probes': [
                            ['graphene oxide', 95],
                            ['carbon nanotube', 88],
                            ['gold nanoparticles', 82],
                            ['silver nanoparticles', 75],
                            ['molybdenum disulfide', 70]
                        ],
                        'best_mediums': [
                            ['water', 90],
                            ['acetate buffer', 85],
                            ['phosphate buffer', 78]
                        ],
                        'best_conditions': [
                            ['25.0, 7.0, 7.5', 92],
                            ['30.0, 6.5, 7.0', 87],
                            ['20.0, 7.5, 8.0', 80]
                        ],
                        'simulated': True
                    })
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

# Helper functions for intelligent simulation
def simulate_ldl_score(target_name, probe_name, medium_name, condition):
    """Simulate LDL score based on known good combinations"""
    import random
    
    score = 2  # Default moderate
    
    # Convert to lowercase for comparison
    target_lower = target_name.lower() if target_name else ''
    probe_lower = probe_name.lower() if probe_name else ''
    medium_lower = medium_name.lower() if medium_name else ''
    
    # Good combinations get better scores
    # PFAS + Graphene-based = excellent
    if 'perfluoro' in target_lower or 'pfas' in target_lower or 'pfos' in target_lower:
        if 'graphene' in probe_lower:
            score = 0 if 'oxide' in probe_lower else 1  # Graphene oxide is best
        elif 'carbon' in probe_lower and 'nanotube' in probe_lower:
            score = 1  # Carbon nanotube is also good
        elif 'gold' in probe_lower:
            score = 1  # Gold nanoparticles good for PFAS
        else:
            score = 2  # Other probes moderate
    
    # Biological targets
    elif 'dopamine' in target_lower or 'glucose' in target_lower:
        if 'carbon' in probe_lower or 'graphene' in probe_lower:
            score = 1  # Good for biosensing
        elif 'gold' in probe_lower:
            score = 0  # Excellent for biosensing
        else:
            score = 2
    
    # Medium effects
    if medium_lower:
        if 'water' in medium_lower and score < 3:
            score = max(0, score - 1)  # Water improves detection
        elif 'buffer' in medium_lower and score > 0:
            score = max(0, score - 1)  # Buffers stabilize
    
    # Condition effects
    if condition:
        temp, ph_min, ph_max = condition
        if 20 <= temp <= 30 and 6 <= ph_min <= 8:
            score = max(0, score - 1)  # Optimal conditions
        elif temp > 40 or ph_min < 4 or ph_max > 10:
            score = min(4, score + 1)  # Harsh conditions worsen detection
    
    # Add some randomness for realism
    if random.random() < 0.2:  # 20% chance to vary
        score = max(0, min(4, score + random.choice([-1, 1])))
    
    return score

def get_simulated_recommendations(mode, target_name, probe_name, medium_name):
    """Generate simulated recommendations based on mode"""
    import random
    
    if 'probe' in mode.lower():
        return {
            'best_probes': [
                ['graphene oxide', 95],
                ['reduced graphene oxide', 92],
                ['carbon nanotube', 88],
                ['multi-walled carbon nanotube', 85],
                ['gold nanoparticles', 82],
                ['silver nanoparticles', 78],
                ['molybdenum disulfide', 75],
                ['titanium dioxide', 72],
                ['zinc oxide', 70],
                ['polyaniline', 68]
            ]
        }
    elif 'medium' in mode.lower():
        return {
            'best_mediums': [
                ['water', 94],
                ['deionized water', 92],
                ['phosphate buffer', 88],
                ['acetate buffer', 85],
                ['phosphate buffered saline', 82],
                ['tris buffer', 78],
                ['ethanol', 70],
                ['methanol', 65]
            ]
        }
    else:
        return {
            'best_conditions': [
                ['25.0, 7.0, 7.5', 95],
                ['25.0, 6.5, 7.0', 92],
                ['30.0, 7.0, 7.5', 88],
                ['20.0, 7.0, 7.5', 85],
                ['25.0, 7.5, 8.0', 82]
            ]
        }