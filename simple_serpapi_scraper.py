"""
Simple SerpAPI News Scraper
Searches BBC, Reuters, and TechCrunch for news articles
"""

import os
import json
from datetime import datetime
from serpapi import GoogleSearch
from typing import List, Dict

# Configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "your_serpapi_key_here")

# Target news sites
NEWS_SITES = {
    "bbc": "site:bbc.com OR site:bbc.co.uk",
    "reuters": "site:reuters.com",
    "techcrunch": "site:techcrunch.com"
}


def search_news(query: str, num_results: int = 10) -> Dict[str, List]:
    """
    Main scraping function that searches 3 news sites for a given query
    
    Args:
        query: Search query string
        num_results: Number of results per site (default 10)
    
    Returns:
        Dictionary with results from each news source
    """
    
    if SERPAPI_KEY == "your_serpapi_key_here":
        return {"error": "Please set your SERPAPI_KEY environment variable"}
    
    all_results = {
        "bbc": [],
        "reuters": [],
        "techcrunch": [],
        "query": query,
        "timestamp": datetime.now().isoformat()
    }
    
    # Search each news site
    for site_name, site_filter in NEWS_SITES.items():
        print(f"Searching {site_name.upper()}...")
        
        # Build search query with site filter
        search_query = f"{query} {site_filter}"
        
        params = {
            "q": search_query,
            "num": num_results,
            "api_key": SERPAPI_KEY,
            "engine": "google"
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Extract organic results
            if "organic_results" in results:
                for result in results["organic_results"]:
                    article = {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "date": result.get("date", ""),
                        "source": site_name
                    }
                    all_results[site_name].append(article)
                    
            print(f"  Found {len(all_results[site_name])} articles")
            
        except Exception as e:
            print(f"  Error searching {site_name}: {e}")
            all_results[site_name] = {"error": str(e)}
    
    return all_results


def search_and_save(query: str, filename: str = None) -> Dict:
    """
    Search news and optionally save to file
    
    Args:
        query: Search query
        filename: Optional filename to save results (JSON)
    
    Returns:
        Search results dictionary
    """
    
    # Perform search
    results = search_news(query)
    
    # Save to file if filename provided
    if filename:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {filename}")
    
    return results


def get_all_articles(results: Dict) -> List[Dict]:
    """
    Extract all articles from results dictionary into a flat list
    
    Args:
        results: Results from search_news()
    
    Returns:
        Flat list of all articles
    """
    
    all_articles = []
    
    for site in ["bbc", "reuters", "techcrunch"]:
        if site in results and isinstance(results[site], list):
            all_articles.extend(results[site])
    
    return all_articles


def filter_by_keywords(results: Dict, keywords: List[str]) -> Dict:
    """
    Filter results to only include articles containing specific keywords
    
    Args:
        results: Results from search_news()
        keywords: List of keywords to filter by
    
    Returns:
        Filtered results dictionary
    """
    
    filtered = {
        "bbc": [],
        "reuters": [],
        "techcrunch": [],
        "query": results.get("query", ""),
        "timestamp": results.get("timestamp", ""),
        "filter_keywords": keywords
    }
    
    for site in ["bbc", "reuters", "techcrunch"]:
        if site in results and isinstance(results[site], list):
            for article in results[site]:
                # Check if any keyword appears in title or snippet
                text = f"{article.get('title', '')} {article.get('snippet', '')}".lower()
                if any(keyword.lower() in text for keyword in keywords):
                    filtered[site].append(article)
    
    return filtered


def print_summary(results: Dict):
    """
    Print a nice summary of search results
    
    Args:
        results: Results from search_news()
    """
    
    print("\n" + "="*60)
    print(f"SEARCH RESULTS for: {results.get('query', 'Unknown')}")
    print("="*60)
    
    total_articles = 0
    
    for site in ["bbc", "reuters", "techcrunch"]:
        if site in results and isinstance(results[site], list):
            count = len(results[site])
            total_articles += count
            print(f"\n{site.upper()}: {count} articles")
            
            # Show first 3 titles from each source
            for i, article in enumerate(results[site][:3], 1):
                print(f"  {i}. {article['title'][:70]}...")
    
    print(f"\nTOTAL: {total_articles} articles found")
    print(f"Timestamp: {results.get('timestamp', 'Unknown')}")
    print("="*60)


# Example usage function that other scripts can call
def get_news_about(topic: str) -> Dict:
    """
    Simple function that other scripts can import and use
    
    Example:
        from simple_serpapi_scraper import get_news_about
        results = get_news_about("artificial intelligence")
    """
    return search_news(topic)


def main():
    """
    Main function for standalone execution
    """
    
    print("Simple News Scraper (BBC, Reuters, TechCrunch)")
    print("-" * 40)
    
    # Get query from user
    query = input("Enter search query: ").strip()
    
    if not query:
        print("No query provided. Using default: 'technology'")
        query = "technology"
    
    # Search
    print(f"\nSearching for: {query}")
    results = search_news(query)
    
    # Print summary
    print_summary(results)
    
    # Ask if user wants to save
    save = input("\nSave results to file? (y/n): ").lower()
    if save == 'y':
        filename = f"news_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {filename}")
    
    # Example of filtering
    print("\nExample: Filtering for specific keywords...")
    keywords = ["AI", "artificial", "technology"]
    filtered = filter_by_keywords(results, keywords)
    
    filtered_count = sum(len(filtered[site]) for site in ["bbc", "reuters", "techcrunch"] 
                        if isinstance(filtered[site], list))
    print(f"Articles containing {keywords}: {filtered_count}")
    
    return results


if __name__ == "__main__":
    # Run standalone
    main()
