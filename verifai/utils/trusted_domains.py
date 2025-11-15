"""Utility to load and display trusted domains from debank-sources.json"""
import json
from pathlib import Path
from typing import List


def load_trusted_domains() -> List[str]:
    """
    Load trusted domains from debank-sources.json
    
    Returns:
        List of trusted domain names
    """
    try:
        # Find the data directory
        current_dir = Path(__file__).parent.parent.parent
        sources_file = current_dir / "data" / "debank-sources.json"
        
        with open(sources_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract all website URLs from the JSON
        domains = []
        
        def extract_websites(obj):
            """Recursively extract website URLs from nested JSON"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "website" and isinstance(value, str):
                        # Clean domain name
                        domain = value.replace("https://", "").replace("http://", "").split("/")[0]
                        domains.append(domain)
                    elif isinstance(value, (dict, list)):
                        extract_websites(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_websites(item)
        
        extract_websites(data)
        return list(set(domains))  # Remove duplicates
    except Exception as e:
        raise ValueError(f"Error loading trusted domains: {e}")


def display_trusted_domains():
    """Display all trusted domains"""
    domains = load_trusted_domains()
    print(f"Found {len(domains)} trusted domains:")
    for domain in sorted(domains):
        print(f"  - {domain}")
    return domains


if __name__ == "__main__":
    display_trusted_domains()

