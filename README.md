# 🕷️ SerpAPI Web Scraping System with RAG Chatbot

A comprehensive web scraping solution using SerpAPI that avoids SSL certificate errors, includes a RAG chatbot, automated scheduling, and a modern web interface.

## 🌟 Features

### Core Scraping
- **No SSL Errors**: Uses SerpAPI to bypass SSL certificate issues
- **Multiple Search Types**: Google, News, Scholar, Images
- **Caching System**: Intelligent caching to reduce API calls
- **Database Storage**: SQLite database for persistent storage
- **Bulk Operations**: Parallel scraping for multiple queries

### RAG Chatbot
- **Context-Aware Responses**: Uses scraped data to answer questions
- **Vector Search**: FAISS-based similarity search
- **Auto-Indexing**: Automatically indexes new scraped content
- **Web & CLI Interface**: Both command-line and Streamlit interfaces

### Monitoring & Scheduling
- **Automated Tasks**: Schedule recurring scraping tasks
- **Performance Metrics**: Track success rates and execution times
- **Multi-Channel Notifications**: Email, Slack, Discord alerts
- **System Monitoring**: CPU, memory, and disk usage tracking

### Web Interface
- **Modern UI**: Tailwind CSS powered responsive interface
- **Real-time Search**: Instant web scraping from the browser
- **Chat Integration**: Built-in chatbot interface
- **Data Management**: Export, statistics, and visualization
- **Task Scheduler**: Visual task management

## 📋 Prerequisites

- Python 3.8+
- SerpAPI Account (Get API key from https://serpapi.com/)
- Optional: Email/Slack/Discord for notifications

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the files
mkdir serpapi-scraper
cd serpapi-scraper

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (one-time setup)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Configuration

```bash
# Set your SerpAPI key
export SERPAPI_KEY="your_serpapi_key_here"

# Or create a .env file
echo "SERPAPI_KEY=your_serpapi_key_here" > .env

# Copy and customize configuration
cp config_template.py config.py
# Edit config.py with your settings
```

### 3. Basic Usage

#### Command Line Scraping
```python
from serpapi_scraper import SerpAPIScraper
import os

# Initialize
api_key = os.getenv("SERPAPI_KEY")
scraper = SerpAPIScraper(api_key)

# Simple search
results = scraper.search_google("artificial intelligence 2024")

# News search
news = scraper.search_news("technology breakthroughs")

# Academic search
papers = scraper.search_scholar("machine learning")

# Bulk search
queries = ["Python", "Data Science", "AI"]
bulk_results = scraper.bulk_search(queries)

# Export data
scraper.export_to_json(results, "output.json")
scraper.export_to_csv(results, "output.csv")
```

#### Web Interface
```bash
# Start the Flask web interface
python web_interface.py

# Open browser to http://localhost:5000
```

#### RAG Chatbot
```bash
# CLI mode
python rag_chatbot.py

# Web mode (Streamlit)
streamlit run rag_chatbot.py web
```

#### Monitoring & Scheduler
```bash
# Start the monitoring system
python monitor_scheduler.py
```

## 📚 Detailed Usage

### Scraping Examples

```python
from serpapi_scraper import SerpAPIScraper
import os

api_key = os.getenv("SERPAPI_KEY")
scraper = SerpAPIScraper(api_key)

# 1. Search with location filter
results = scraper.search_google(
    "restaurants",
    location="New York, NY"
)

# 2. Time-filtered news search
recent_news = scraper.search_news(
    "artificial intelligence",
    time_filter="qdr:d"  # Last day
)

# 3. Academic search with year filter
papers = scraper.search_scholar(
    "quantum computing",
    year_from=2023
)

# 4. Enrich with full content extraction
enriched = scraper.enrich_with_full_content(results[:5])

# 5. Get analysis
from serpapi_scraper import SmartAnalyzer
analyzer = SmartAnalyzer()
stats = analyzer.get_summary_stats(results)
duplicates = analyzer.find_duplicates(results)
```

### Chatbot Usage

```python
from rag_chatbot import RAGChatbot

# Initialize chatbot
chatbot = RAGChatbot()

# Ask questions about scraped data
response = chatbot.generate_response("What are the latest AI trends?")
print(response["response"])

# Check sources
for source in response["sources"]:
    print(f"- {source['title']}: {source['url']}")

# Update index with new data
new_data = [{"title": "...", "content": "...", ...}]
chatbot.update_index(new_data)
```

### Scheduling Tasks

```python
from monitor_scheduler import TaskScheduler

scheduler = TaskScheduler(api_key)

# Add daily news task
scheduler.add_task(
    name="daily_tech_news",
    query="technology news today",
    schedule_str="09:00",  # 9 AM daily
    search_type="news"
)

# Add recurring task
scheduler.add_recurring_task(
    name="ai_monitoring",
    query="artificial intelligence breakthroughs",
    interval_minutes=60  # Every hour
)

# Start scheduler
scheduler.start()

# Run specific task now
scheduler.run_task_now("daily_tech_news")

# Get status
status = scheduler.get_status()
print(f"Success rate: {status['metrics']['success_rate']}%")
```

## 🎨 Web Interface Features

### Main Dashboard
- **Search Panel**: Instant web search with multiple engines
- **Results Display**: Clean, organized search results
- **Export Options**: JSON, CSV, Markdown formats

### RAG Chat
- **Interactive Chat**: Ask questions about your data
- **Source Attribution**: See which documents inform responses
- **Context Toggle**: Choose whether to use scraped data

### Data Management
- **Statistics**: Total documents, sources, last update
- **Recent Data**: Table view of latest scraped content
- **Actions**: Rebuild index, clear cache

### Task Scheduler
- **Visual Status**: See running tasks and success rates
- **Add Tasks**: Create new scheduled scraping tasks
- **Manual Execution**: Run tasks on-demand

## 🔧 Configuration Options

Edit `config.py` for:

- **API Settings**: SerpAPI key and rate limits
- **Cache Settings**: TTL and size limits
- **Export Formats**: Default export options
- **Notification Channels**: Email, Slack, Discord
- **Scheduled Tasks**: Pre-configured scraping jobs
- **RAG Settings**: Embedding model and chunk size

## 📊 Database Schema

### scraped_data table
- `id`: Primary key
- `title`: Article/page title
- `url`: Source URL
- `snippet`: Short description
- `source`: Website/publisher
- `date`: Publication date
- `full_content`: Extended content (if extracted)
- `metadata`: JSON additional data
- `query`: Original search query
- `scrape_timestamp`: When scraped

### search_history table
- `id`: Primary key
- `query`: Search query
- `engine`: Search engine used
- `results_count`: Number of results
- `search_timestamp`: When searched

## 🚨 Error Handling

The system includes comprehensive error handling:

- **API Errors**: Automatic retries with exponential backoff
- **Network Issues**: Graceful degradation and caching
- **Database Errors**: Transaction rollback and logging
- **Content Extraction**: Multiple fallback methods

## 📈 Performance Tips

1. **Use Caching**: Enable cache to reduce API calls
2. **Batch Operations**: Use bulk search for multiple queries
3. **Parallel Processing**: Enable parallel mode for speed
4. **Selective Enrichment**: Only extract full content when needed
5. **Index Management**: Rebuild index periodically

## 🔒 Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- Implement rate limiting for production use
- Sanitize user inputs in web interface
- Use HTTPS in production deployments

## 📝 Common Issues

### SSL Certificate Errors
This system uses SerpAPI specifically to avoid SSL issues. If you still encounter problems:
- Verify your API key is correct
- Check your network connection
- Ensure SerpAPI service is accessible

### Memory Issues with Large Datasets
- Limit the number of results per search
- Use pagination for large exports
- Clear cache periodically
- Implement data retention policies

### Slow Search Performance
- Enable caching
- Use parallel processing
- Reduce enrichment operations
- Optimize database queries

## 🤝 Contributing

Feel free to extend this system with:
- Additional search engines
- New export formats
- Enhanced NLP features
- Custom data processors
- UI improvements

## 📄 License

This project is provided as-is for educational and commercial use.

## 🆘 Support

For issues:
1. Check the logs in `scraper.log` and `monitor.log`
2. Verify your API key and configuration
3. Ensure all dependencies are installed
4. Check SerpAPI documentation at https://serpapi.com/docs

## 🎯 Next Steps

1. **Production Deployment**:
   - Use environment variables for all secrets
   - Set up proper logging infrastructure
   - Implement user authentication
   - Add rate limiting and quotas

2. **Advanced Features**:
   - Sentiment analysis on scraped content
   - Entity extraction and knowledge graphs
   - Custom ML models for classification
   - Integration with external APIs

3. **Scaling**:
   - Use Redis for caching
   - PostgreSQL for larger datasets
   - Celery for distributed task processing
   - Docker for containerization

---

**Remember**: Always respect robots.txt, rate limits, and copyright when scraping. This tool uses SerpAPI which handles these concerns, but be mindful when extracting full content from websites.
