import json
from itertools import combinations

def generate_graph_data(data):
    edges = []
    
    def extract_year(date_string):
        return date_string.split('-')[0]
    
    def extract_year_month(date_string):
        return date_string[:7]

    for item1, item2 in combinations(data, 2):
        source = item1['subject']
        target = item2['subject']
        
        if item1['all'][0]['creator'] == item2['all'][0]['creator']:
            edges.append({
                "source": source,
                "target": target,
                "relationship": f"Same Creator-{item1['all'][0]['creator']}"
            })
        
        if item1['all'][0]['Thrust'] == item2['all'][0]['Thrust']:
            edges.append({
                "source": source,
                "target": target,
                "relationship": f"Same Thrust-{item1['all'][0]['Thrust']}"
            })
        
        common_types = set(item1['all'][0]['Data Type']) & set(item2['all'][0]['Data Type'])
        for data_type in common_types:
            edges.append({
                "source": source,
                "target": target,
                "relationship": f"Same Data Type-{data_type}"
            })
        
        common_topics = set(item1['all'][0]['Related Topic']) & set(item2['all'][0]['Related Topic'])
        for topic in common_topics:
            edges.append({
                "source": source,
                "target": target,
                "relationship": f"Same Related Topic-{topic}"
            })
        
        if extract_year(item1['all'][0]['date']) == extract_year(item2['all'][0]['date']):
            edges.append({
                "source": source,
                "target": target,
                "relationship": f"Same Year-{extract_year(item1['all'][0]['date'])}"
            })
        
        if extract_year_month(item1['all'][0]['date']) == extract_year_month(item2['all'][0]['date']):
            edges.append({
                "source": source,
                "target": target,
                "relationship": f"Same Year Month-{extract_year_month(item1['all'][0]['date'])}"
            })
        
        common_pis = set(item1['all'][0]['PI Affiliated']) & set(item2['all'][0]['PI Affiliated'])
        for pi in common_pis:
            edges.append({
                "source": source,
                "target": target,
                "relationship": f"Same PI Affiliated-{pi}"
            })
        
        if item1['all'][0]['Document Format'] == item2['all'][0]['Document Format']:
            edges.append({
                "source": source,
                "target": target,
                "relationship": f"Same Document Format-{item1['all'][0]['Document Format']}"
            })
        
        common_tags = set(item1['all'][0]['Data Tags']) & set(item2['all'][0]['Data Tags'])
        for tag in common_tags:
            edges.append({
                "source": source,
                "target": target,
                "relationship": f"Same Data Tags-{tag}"
            })

    return {
        "ingest_type": "GMetaList",
        "ingest_data": {
            "edges": edges,
            "gmeta": data
        }
    }

# Load the JSON data
with open('total_search_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Generate the graph data
graph_data = generate_graph_data(data)

# Save the result to graph.json
with open('graph_new.json', 'w', encoding='utf-8') as f:
    json.dump(graph_data, f, indent=2, ensure_ascii=False)

print(f"Generated graph data with {len(graph_data['ingest_data']['edges'])} edges and {len(graph_data['ingest_data']['gmeta'])} nodes, saved to graph_new.json")