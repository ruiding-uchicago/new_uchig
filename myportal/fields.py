import os
from urllib.parse import urlsplit, urlunsplit, urlencode
import datetime
from typing import List, Mapping, Any

def title(result):
    """The title for this Globus Search subject"""
    return result[0]["Title"]


def globus_app_link(result):
    """A Globus Webapp link for the transfer/sync button on the detail page"""
    url = result[0]["url"]
    parsed = urlsplit(url)
    query_params = {
        "origin_id": parsed.netloc,
        "origin_path": os.path.dirname(parsed.path),
    }
    return urlunsplit(
        ("https", "app.globus.org", "file-manager", urlencode(query_params), "")
    )


def https_url(result):
    """Add a direct download link to files over HTTPS"""
    path = urlsplit(result[0]["url"]).path
    return urlunsplit(("https", "g-71c9e9.10bac.8443.data.globus.org", path, "", ""))

def search_highlights(result: List[Mapping[str, Any]]) -> List[Mapping[str, dict]]:
    """Prepare the most useful pieces of information for users on the search results page."""
    search_highlights = list()
    for name in ["creator", "Data Tags",  "PI Affiliated","date", "Thrust"]:
        value = result[0].get(name)
        value_type = "str"

        # Parse a date if it's a date. All dates expected isoformat
        if name == "date":
            value = datetime.datetime.fromisoformat(value)
            value_type = "date"
        elif name == "Data Tags":
            value = ", ".join(value)

        # Add the value to the list
        search_highlights.append(
            {
                "name": name,
                "title": name.capitalize(),
                "value": value,
                "type": value_type,
            }
        )
    return search_highlights

def document_format(result):
    """Extract the document format from the result."""
    return result[0]['Document Format']
def data_type(result):
    """Extract the data type from the result."""
    return result[0]['Data Type']
def abstract_description(result):
    """Extract the abstract description from the result."""
    return result[0]['Abstract Description']
def outer_link(result):
    """Extract the outer link from the result."""
    return result[0]['Outer Link']
def related_topic(result):
    """Extract the outer link from the result."""
    return result[0]['Related Topic']
def dc(result):
    """Render metadata in datacite format, Must confrom to the datacite spec"""
    date = datetime.datetime.fromisoformat(result[0]['date'])
    return {
        "formats": ["text/plain"],
        "creators": [{"creatorName": result[0]['creator']}],
        "contributors": [{"contributorName": result[0]['creator']}],
        "subjects": [{"subject": s for s in result[0]['Data Tags']}],
        "publicationYear": date.year,
        "publisher": "Organization",
        "dates": [{"date": date,
                  "dateType": "Created"}],
        "titles": [{"title": result[0]['Title']}],
        "version": "1",
        "resourceType": {
            "resourceTypeGeneral": "Dataset",
            "resourceType": "Dataset"
        }
    }


def project_metadata(result):
    """Render any project-specific metadata for this project. Does not conform to
    a spec and can be of any type, although values should be generally human readable."""
    project_metadata_names = ['times_accessed', 'original_collection_name']
    return {k: v for k, v in result[0].items() if k in project_metadata_names}