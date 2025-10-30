# Configuration file for SerpAPI Scraper System
# Copy this to config.py and update with your settings

import os
from typing import Dict, List

# API Configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "your_serpapi_key_here")

# Database Configuration
DATABASE_CONFIG = {
    "path": "scraped_data.db",
    "backup_enabled": True,
    "backup_interval_hours": 24
}

# Cache Configuration
CACHE_CONFIG = {
    "directory": "cache",
    "expire_hours": 24,
    "max_size_mb": 100
}

# Scraping Configuration
SCRAPING_CONFIG = {
    "rate_limit_delay": 1,  # seconds between requests
    "max_parallel_requests": 3,
    "timeout": 30,  # seconds
    "retry_attempts": 3,
    "user_agents": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    ]
}

# Content Extraction Configuration
EXTRACTION_CONFIG = {
    "max_content_length": 10000,  # characters
    "extract_images": True,
    "extract_links": True,
    "clean_html": True
}

# Export Configuration
EXPORT_CONFIG = {
    "output_directory": "exports",
    "formats": ["json", "csv", "markdown", "html"],
    "compress": True  # Create zip files for large exports
}

# News Sources Configuration
NEWS_SOURCES = {
    "tech": [
        "TechCrunch",
        "The Verge",
        "Wired",
        "Ars Technica",
        "Hacker News"
    ],
    "general": [
        "BBC",
        "CNN",
        "Reuters",
        "The Guardian",
        "Associated Press"
    ],
    "business": [
        "Bloomberg",
        "Financial Times",
        "Wall Street Journal",
        "Forbes",
        "Business Insider"
    ],
    "science": [
        "Nature",
        "Science Magazine",
        "Scientific American",
        "New Scientist",
        "MIT Technology Review"
    ]
}

# Search Queries Templates
SEARCH_TEMPLATES = {
    "trending": "{topic} trending news {year}",
    "analysis": "{topic} analysis expert opinion",
    "research": "{topic} research papers studies",
    "tutorial": "{topic} tutorial guide how to",
    "comparison": "{topic} vs {alternative} comparison"
}

# Monitoring Configuration
MONITORING_CONFIG = {
    "enable_logging": True,
    "log_level": "INFO",
    "log_file": "scraper.log",
    "alert_email": None,  # Set email for alerts
    "webhook_url": None,  # Set webhook for notifications
    "metrics_enabled": True
}

# Scheduled Tasks Configuration
SCHEDULED_TASKS = [
    {
        "name": "daily_tech_news",
        "schedule": "0 9 * * *",  # 9 AM daily
        "query": "technology news today",
        "search_type": "news",
        "auto_export": True
    },
    {
        "name": "weekly_ai_research",
        "schedule": "0 10 * * MON",  # 10 AM every Monday
        "query": "artificial intelligence research papers",
        "search_type": "scholar",
        "auto_export": True
    }
]

# RAG System Configuration
RAG_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "vector_db_path": "vector_store",
    "similarity_threshold": 0.7
}

# Web Interface Configuration
WEB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": True,
    "secret_key": "your-secret-key-here",
    "enable_api": True,
    "api_rate_limit": "100/hour"
}

# Data Processing Configuration
PROCESSING_CONFIG = {
    "remove_duplicates": True,
    "similarity_threshold": 0.85,
    "min_content_length": 100,
    "max_content_length": 50000,
    "language_filter": ["en"],  # Filter by language codes
    "keyword_filters": {
        "exclude": [],  # Keywords to exclude
        "include": []   # Keywords to require
    }
}

# Notification Configuration
NOTIFICATION_CONFIG = {
    "email": {
        "enabled": False,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "",
        "password": "",
        "recipients": []
    },
    "slack": {
        "enabled": False,
        "webhook_url": "",
        "channel": "#scraping-alerts"
    },
    "discord": {
        "enabled": False,
        "webhook_url": ""
    }
}

# Custom Headers for Different Sources
CUSTOM_HEADERS = {
    "default": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
}

# Quality Control Configuration
QUALITY_CONFIG = {
    "min_title_length": 10,
    "min_snippet_length": 50,
    "require_date": False,
    "require_source": True,
    "validate_urls": True,
    "check_broken_links": False
}

# Advanced Features Toggle
FEATURES = {
    "auto_translate": False,
    "sentiment_analysis": False,
    "entity_extraction": False,
    "keyword_extraction": True,
    "auto_summarize": True,
    "screenshot_capture": False,
    "pdf_generation": True
}

def validate_config() -> Dict[str, bool]:
    """Validate configuration settings"""
    validation = {}
    
    # Check API key
    validation["api_key"] = SERPAPI_KEY != "your_serpapi_key_here"
    
    # Check paths
    validation["paths"] = all([
        os.access(".", os.W_OK),  # Can write to current directory
    ])
    
    # Check dependencies
    try:
        import serpapi
        validation["serpapi"] = True
    except ImportError:
        validation["serpapi"] = False
    
    return validation

if __name__ == "__main__":
    # Validate configuration when run directly
    validation = validate_config()
    print("Configuration Validation:")
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    if not all(validation.values()):
        print("\n⚠️  Some configuration items need attention!")
    else:
        print("\n✅ Configuration is valid!")
