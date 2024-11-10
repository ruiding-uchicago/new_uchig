import json
from itertools import combinations
from collections import defaultdict

def generate_graph_data(data):
    edges = []
    
    # Pre-process data into dictionaries for faster lookups
    creators = defaultdict(list)
    thrusts = defaultdict(list)
    data_types = defaultdict(list)
    topics = defaultdict(list)
    years = defaultdict(list)
    year_months = defaultdict(list)
    pis = defaultdict(list)
    document_formats = defaultdict(list)
    data_tags = defaultdict(list)

    for item in data:
        subject = item['subject']
        all_data = item['all'][0]
        
        creators[all_data['creator']].append(subject)
        thrusts[all_data['Thrust']].append(subject)
        for dt in all_data['Data Type']:
            data_types[dt].append(subject)
        for topic in all_data['Related Topic']:
            topics[topic].append(subject)
        
        year = all_data['date'][:4]
        years[year].append(subject)
        year_month = all_data['date'][:7]
        year_months[year_month].append(subject)
        
        for pi in all_data['PI Affiliated']:
            pis[pi].append(subject)
        document_formats[all_data['Document Format']].append(subject)
        for tag in all_data['Data Tags']:
            data_tags[tag].append(subject)

    # Generate edges
    def add_edges(dictionary, relationship_prefix):
        for key, subjects in dictionary.items():
            for source, target in combinations(subjects, 2):
                edges.append({
                    "source": source,
                    "target": target,
                    "relationship": f"{relationship_prefix}-{key}"
                })

    add_edges(creators, "Same Creator")
    add_edges(thrusts, "Same Thrust")
    add_edges(data_types, "Same Data Type")
    add_edges(topics, "Same Related Topic")
    add_edges(years, "Same Year")
    add_edges(year_months, "Same Year Month")
    add_edges(pis, "Same PI Affiliated")
    add_edges(document_formats, "Same Document Format")
    add_edges(data_tags, "Same Data Tags")

    return {
        "ingest_type": "GMetaList",
        "ingest_data": {
            "edges": edges,
            "gmeta": data
        }
    }