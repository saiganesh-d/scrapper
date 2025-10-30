"""
Enhanced Web Scraping System using SerpAPI
Features:
- No SSL certificate errors
- Multiple search engines support
- Advanced data processing
- Caching and rate limiting
- Export to multiple formats
"""

import os
import json
import time
import hashlib
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ScrapedData:
    """Data class for scraped content"""
    title: str
    url: str
    snippet: str
    source: str
    date: str
    full_content: Optional[str] = None
    metadata: Optional[Dict] = None
    scrape_timestamp: str = None
    
    def __post_init__(self):
        if not self.scrape_timestamp:
            self.scrape_timestamp = datetime.now().isoformat()


class CacheManager:
    """Manages caching of scraped data"""
    
    def __init__(self, cache_dir: str = "cache", expire_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.expire_hours = expire_hours
        
    def _get_cache_key(self, query: str, params: dict) -> str:
        """Generate unique cache key"""
        cache_string = f"{query}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, query: str, params: dict) -> Optional[Any]:
        """Get cached data if not expired"""
        cache_key = self._get_cache_key(query, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time < timedelta(hours=self.expire_hours):
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Cache hit for query: {query}")
                        return pickle.load(f)
                except Exception as e:
                    logger.error(f"Cache read error: {e}")
        return None
    
    def set(self, query: str, params: dict, data: Any):
        """Save data to cache"""
        cache_key = self._get_cache_key(query, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                logger.info(f"Cached data for query: {query}")
        except Exception as e:
            logger.error(f"Cache write error: {e}")


class DatabaseManager:
    """Manages SQLite database for storing scraped data"""
    
    def __init__(self, db_path: str = "scraped_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scraped_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                url TEXT UNIQUE,
                snippet TEXT,
                source TEXT,
                date TEXT,
                full_content TEXT,
                metadata TEXT,
                scrape_timestamp TIMESTAMP,
                query TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                engine TEXT,
                results_count INTEGER,
                search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_data(self, data: ScrapedData, query: str):
        """Save scraped data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO scraped_data 
                (title, url, snippet, source, date, full_content, metadata, scrape_timestamp, query)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.title,
                data.url,
                data.snippet,
                data.source,
                data.date,
                data.full_content,
                json.dumps(data.metadata) if data.metadata else None,
                data.scrape_timestamp,
                query
            ))
            conn.commit()
            logger.info(f"Saved to database: {data.title}")
        except Exception as e:
            logger.error(f"Database save error: {e}")
        finally:
            conn.close()
    
    def get_recent_data(self, hours: int = 24) -> List[Dict]:
        """Get recent scraped data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since = datetime.now() - timedelta(hours=hours)
        cursor.execute('''
            SELECT * FROM scraped_data 
            WHERE created_at > ? 
            ORDER BY created_at DESC
        ''', (since.isoformat(),))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results


class ContentExtractor:
    """Extract full content from URLs using multiple methods"""
    
    @staticmethod
    def extract_with_newspaper(url: str) -> Optional[str]:
        """Extract content using newspaper3k library"""
        try:
            from newspaper import Article
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except:
            return None
    
    @staticmethod
    def extract_with_beautifulsoup(url: str) -> Optional[str]:
        """Extract content using BeautifulSoup"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:5000]  # Limit to 5000 chars
        except Exception as e:
            logger.error(f"BeautifulSoup extraction error for {url}: {e}")
            return None
    
    @staticmethod
    def extract_content(url: str) -> Optional[str]:
        """Try multiple methods to extract content"""
        # Try newspaper first
        content = ContentExtractor.extract_with_newspaper(url)
        if content:
            return content
        
        # Fall back to BeautifulSoup
        return ContentExtractor.extract_with_beautifulsoup(url)


class SerpAPIScraper:
    """Main scraper class using SerpAPI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.cache = CacheManager()
        self.db = DatabaseManager()
        self.content_extractor = ContentExtractor()
        self.rate_limit_delay = 1  # seconds between requests
        
    def search_google(self, query: str, num_results: int = 10, 
                      location: str = None, time_filter: str = None) -> List[ScrapedData]:
        """Search Google using SerpAPI"""
        
        params = {
            "q": query,
            "num": num_results,
            "api_key": self.api_key,
            "engine": "google"
        }
        
        if location:
            params["location"] = location
        if time_filter:  # qdr:d (day), qdr:w (week), qdr:m (month), qdr:y (year)
            params["tbs"] = time_filter
        
        # Check cache first
        cached_data = self.cache.get(query, params)
        if cached_data:
            return cached_data
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            scraped_data = []
            
            # Process organic results
            if "organic_results" in results:
                for result in results["organic_results"]:
                    data = ScrapedData(
                        title=result.get("title", ""),
                        url=result.get("link", ""),
                        snippet=result.get("snippet", ""),
                        source=result.get("source", ""),
                        date=result.get("date", ""),
                        metadata={
                            "position": result.get("position"),
                            "displayed_link": result.get("displayed_link")
                        }
                    )
                    scraped_data.append(data)
                    self.db.save_data(data, query)
            
            # Cache the results
            self.cache.set(query, params, scraped_data)
            
            # Log search history
            self._log_search(query, "google", len(scraped_data))
            
            return scraped_data
            
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return []
    
    def search_news(self, query: str, time_filter: str = "qdr:d") -> List[ScrapedData]:
        """Search Google News using SerpAPI"""
        
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google_news",
            "when": "1d"  # Last day by default
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            scraped_data = []
            
            if "news_results" in results:
                for result in results["news_results"]:
                    data = ScrapedData(
                        title=result.get("title", ""),
                        url=result.get("link", ""),
                        snippet=result.get("snippet", ""),
                        source=result.get("source", {}).get("name", ""),
                        date=result.get("date", ""),
                        metadata={
                            "thumbnail": result.get("thumbnail")
                        }
                    )
                    scraped_data.append(data)
                    self.db.save_data(data, query)
            
            self._log_search(query, "google_news", len(scraped_data))
            return scraped_data
            
        except Exception as e:
            logger.error(f"News search error: {e}")
            return []
    
    def search_scholar(self, query: str, year_from: int = None) -> List[ScrapedData]:
        """Search Google Scholar using SerpAPI"""
        
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google_scholar"
        }
        
        if year_from:
            params["as_ylo"] = year_from
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            scraped_data = []
            
            if "organic_results" in results:
                for result in results["organic_results"]:
                    data = ScrapedData(
                        title=result.get("title", ""),
                        url=result.get("link", ""),
                        snippet=result.get("snippet", ""),
                        source=result.get("publication_info", {}).get("summary", ""),
                        date="",
                        metadata={
                            "authors": result.get("publication_info", {}).get("authors", []),
                            "cited_by": result.get("inline_links", {}).get("cited_by", {}).get("total", 0)
                        }
                    )
                    scraped_data.append(data)
                    self.db.save_data(data, query)
            
            self._log_search(query, "google_scholar", len(scraped_data))
            return scraped_data
            
        except Exception as e:
            logger.error(f"Scholar search error: {e}")
            return []
    
    def search_images(self, query: str) -> List[Dict]:
        """Search Google Images using SerpAPI"""
        
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google_images"
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            images = []
            if "images_results" in results:
                for img in results["images_results"][:10]:  # Limit to 10 images
                    images.append({
                        "title": img.get("title", ""),
                        "link": img.get("link", ""),
                        "original": img.get("original", ""),
                        "thumbnail": img.get("thumbnail", "")
                    })
            
            return images
            
        except Exception as e:
            logger.error(f"Image search error: {e}")
            return []
    
    def bulk_search(self, queries: List[str], search_type: str = "google", 
                    parallel: bool = True) -> Dict[str, List[ScrapedData]]:
        """Perform bulk searches with optional parallelization"""
        
        results = {}
        
        if parallel:
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_query = {}
                
                for query in queries:
                    if search_type == "google":
                        future = executor.submit(self.search_google, query)
                    elif search_type == "news":
                        future = executor.submit(self.search_news, query)
                    elif search_type == "scholar":
                        future = executor.submit(self.search_scholar, query)
                    else:
                        continue
                    
                    future_to_query[future] = query
                    time.sleep(self.rate_limit_delay)  # Rate limiting
                
                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        data = future.result()
                        results[query] = data
                        logger.info(f"Completed search for: {query}")
                    except Exception as e:
                        logger.error(f"Error searching '{query}': {e}")
                        results[query] = []
        else:
            for query in queries:
                if search_type == "google":
                    results[query] = self.search_google(query)
                elif search_type == "news":
                    results[query] = self.search_news(query)
                elif search_type == "scholar":
                    results[query] = self.search_scholar(query)
                
                time.sleep(self.rate_limit_delay)
        
        return results
    
    def enrich_with_full_content(self, scraped_data: List[ScrapedData], 
                                 max_items: int = 5) -> List[ScrapedData]:
        """Enrich scraped data with full content extraction"""
        
        enriched = []
        count = 0
        
        for data in scraped_data:
            if count >= max_items:
                enriched.append(data)
                continue
            
            if data.url:
                logger.info(f"Extracting full content from: {data.url}")
                full_content = self.content_extractor.extract_content(data.url)
                if full_content:
                    data.full_content = full_content
                    count += 1
            
            enriched.append(data)
            time.sleep(0.5)  # Be polite to servers
        
        return enriched
    
    def export_to_json(self, data: List[ScrapedData], filename: str = "scraped_data.json"):
        """Export data to JSON file"""
        json_data = [asdict(item) for item in data]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported {len(data)} items to {filename}")
    
    def export_to_csv(self, data: List[ScrapedData], filename: str = "scraped_data.csv"):
        """Export data to CSV file"""
        df = pd.DataFrame([asdict(item) for item in data])
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"Exported {len(data)} items to {filename}")
    
    def export_to_markdown(self, data: List[ScrapedData], filename: str = "scraped_data.md"):
        """Export data to Markdown file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Scraped Data Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for item in data:
                f.write(f"## {item.title}\n\n")
                f.write(f"**URL:** {item.url}\n\n")
                f.write(f"**Source:** {item.source}\n\n")
                f.write(f"**Date:** {item.date}\n\n")
                f.write(f"**Snippet:** {item.snippet}\n\n")
                
                if item.full_content:
                    f.write(f"**Full Content (excerpt):**\n\n")
                    f.write(f"{item.full_content[:500]}...\n\n")
                
                f.write("---\n\n")
        
        logger.info(f"Exported {len(data)} items to {filename}")
    
    def _log_search(self, query: str, engine: str, results_count: int):
        """Log search to database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO search_history (query, engine, results_count)
            VALUES (?, ?, ?)
        ''', (query, engine, results_count))
        
        conn.commit()
        conn.close()


class SmartAnalyzer:
    """Analyze and summarize scraped data"""
    
    @staticmethod
    def get_summary_stats(data: List[ScrapedData]) -> Dict:
        """Get summary statistics"""
        stats = {
            "total_results": len(data),
            "unique_sources": len(set(item.source for item in data if item.source)),
            "date_range": SmartAnalyzer._get_date_range(data),
            "top_sources": SmartAnalyzer._get_top_sources(data),
            "avg_snippet_length": sum(len(item.snippet) for item in data) / len(data) if data else 0
        }
        return stats
    
    @staticmethod
    def _get_date_range(data: List[ScrapedData]) -> str:
        """Get date range of results"""
        dates = [item.date for item in data if item.date]
        if not dates:
            return "No dates available"
        return f"{min(dates)} to {max(dates)}"
    
    @staticmethod
    def _get_top_sources(data: List[ScrapedData], top_n: int = 5) -> List[tuple]:
        """Get top sources by frequency"""
        from collections import Counter
        sources = [item.source for item in data if item.source]
        return Counter(sources).most_common(top_n)
    
    @staticmethod
    def find_duplicates(data: List[ScrapedData]) -> List[List[ScrapedData]]:
        """Find duplicate or similar content"""
        from difflib import SequenceMatcher
        
        duplicates = []
        seen = set()
        
        for i, item1 in enumerate(data):
            if i in seen:
                continue
            
            similar = [item1]
            for j, item2 in enumerate(data[i+1:], i+1):
                if j in seen:
                    continue
                
                # Check title similarity
                similarity = SequenceMatcher(None, item1.title, item2.title).ratio()
                if similarity > 0.8:  # 80% similar
                    similar.append(item2)
                    seen.add(j)
            
            if len(similar) > 1:
                duplicates.append(similar)
                seen.add(i)
        
        return duplicates


def main():
    """Main execution function with example usage"""
    
    # Initialize scraper with your API key
    # Get your API key from: https://serpapi.com/
    API_KEY = os.getenv("SERPAPI_KEY", "your_api_key_here")
    
    if API_KEY == "your_api_key_here":
        logger.error("Please set your SERPAPI_KEY environment variable or update the API_KEY in the code")
        return
    
    scraper = SerpAPIScraper(API_KEY)
    
    # Example 1: Simple Google search
    print("\n=== Google Search Example ===")
    results = scraper.search_google("artificial intelligence trends 2024", num_results=5)
    for item in results[:3]:
        print(f"Title: {item.title}")
        print(f"URL: {item.url}")
        print(f"Snippet: {item.snippet[:100]}...")
        print("-" * 50)
    
    # Example 2: News search
    print("\n=== News Search Example ===")
    news_results = scraper.search_news("technology breakthroughs")
    for item in news_results[:3]:
        print(f"Title: {item.title}")
        print(f"Source: {item.source}")
        print(f"Date: {item.date}")
        print("-" * 50)
    
    # Example 3: Academic search
    print("\n=== Scholar Search Example ===")
    scholar_results = scraper.search_scholar("machine learning", year_from=2023)
    for item in scholar_results[:3]:
        print(f"Title: {item.title}")
        if item.metadata and "cited_by" in item.metadata:
            print(f"Citations: {item.metadata['cited_by']}")
        print("-" * 50)
    
    # Example 4: Bulk search with multiple queries
    print("\n=== Bulk Search Example ===")
    queries = ["Python programming", "Data science", "Web scraping"]
    bulk_results = scraper.bulk_search(queries, search_type="google")
    
    for query, results in bulk_results.items():
        print(f"\nQuery: {query}")
        print(f"Found {len(results)} results")
    
    # Example 5: Enrich with full content
    print("\n=== Content Enrichment Example ===")
    if results:
        enriched = scraper.enrich_with_full_content(results[:2], max_items=2)
        for item in enriched:
            if item.full_content:
                print(f"Title: {item.title}")
                print(f"Content preview: {item.full_content[:200]}...")
                print("-" * 50)
    
    # Example 6: Export data
    print("\n=== Exporting Data ===")
    all_results = results + news_results + scholar_results
    
    # Export to different formats
    scraper.export_to_json(all_results, "search_results.json")
    scraper.export_to_csv(all_results, "search_results.csv")
    scraper.export_to_markdown(all_results, "search_results.md")
    
    # Example 7: Analyze results
    print("\n=== Analysis ===")
    analyzer = SmartAnalyzer()
    stats = analyzer.get_summary_stats(all_results)
    print(f"Total results: {stats['total_results']}")
    print(f"Unique sources: {stats['unique_sources']}")
    print(f"Top sources: {stats['top_sources']}")
    
    # Example 8: Get recent data from database
    print("\n=== Recent Database Entries ===")
    recent_data = scraper.db.get_recent_data(hours=24)
    print(f"Found {len(recent_data)} entries in the last 24 hours")
    
    # Example 9: Find duplicates
    duplicates = analyzer.find_duplicates(all_results)
    if duplicates:
        print(f"\nFound {len(duplicates)} groups of similar content")


if __name__ == "__main__":
    main()
