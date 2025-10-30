"""
Simple Chatbot that uses the News Scraper
Can fetch news and answer questions about current events
"""

import json
import os
from datetime import datetime
from simple_serpapi_scraper import search_news, get_all_articles, filter_by_keywords


class SimpleNewsBot:
    """
    A simple chatbot that fetches and discusses news
    """
    
    def __init__(self):
        self.cached_results = {}
        self.last_query = None
        self.cache_dir = "news_cache"
        
        # Create cache directory
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def fetch_news(self, query: str) -> dict:
        """
        Fetch news using the scraper, with simple caching
        """
        # Check if we already searched this recently
        if query in self.cached_results:
            print("Using cached results...")
            return self.cached_results[query]
        
        # Fetch new results
        print(f"Fetching news about: {query}")
        results = search_news(query)
        
        # Cache results
        self.cached_results[query] = results
        self.last_query = query
        
        # Save to file for persistence
        cache_file = os.path.join(self.cache_dir, f"{query.replace(' ', '_')}.json")
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def answer_question(self, question: str) -> str:
        """
        Answer questions about news topics
        """
        question_lower = question.lower()
        
        # Extract topic from question
        topic = self.extract_topic(question)
        
        # Fetch relevant news
        results = self.fetch_news(topic)
        
        # Get all articles
        all_articles = get_all_articles(results)
        
        if not all_articles:
            return f"Sorry, I couldn't find any news about '{topic}'. Please try a different query."
        
        # Generate response based on question type
        if "how many" in question_lower:
            return self.count_articles(results, topic)
        
        elif "latest" in question_lower or "recent" in question_lower:
            return self.get_latest_news(all_articles, topic)
        
        elif "summary" in question_lower or "summarize" in question_lower:
            return self.summarize_news(all_articles, topic)
        
        elif "which site" in question_lower or "which source" in question_lower:
            return self.compare_sources(results)
        
        else:
            # Default: show top headlines
            return self.get_headlines(all_articles, topic)
    
    def extract_topic(self, question: str) -> str:
        """
        Simple topic extraction from question
        """
        # Remove common question words
        stop_words = ['what', 'is', 'are', 'the', 'news', 'about', 'tell', 'me', 
                     'show', 'latest', 'recent', 'how', 'many', 'which', 'summary',
                     'summarize', 'headlines', 'on', 'for', 'of', 'in']
        
        words = question.lower().split()
        topic_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        if topic_words:
            return ' '.join(topic_words)
        else:
            return "technology"  # Default topic
    
    def count_articles(self, results: dict, topic: str) -> str:
        """
        Count articles from each source
        """
        response = f"📊 Article count for '{topic}':\n\n"
        
        total = 0
        for site in ["bbc", "reuters", "techcrunch"]:
            if site in results and isinstance(results[site], list):
                count = len(results[site])
                total += count
                response += f"• {site.upper()}: {count} articles\n"
        
        response += f"\n📰 Total: {total} articles found"
        return response
    
    def get_latest_news(self, articles: list, topic: str) -> str:
        """
        Get the latest news headlines
        """
        if not articles:
            return "No articles found."
        
        response = f"🔥 Latest news about '{topic}':\n\n"
        
        # Show first 5 articles
        for i, article in enumerate(articles[:5], 1):
            response += f"{i}. {article['title']}\n"
            response += f"   Source: {article['source'].upper()}\n"
            if article.get('snippet'):
                response += f"   {article['snippet'][:100]}...\n"
            response += "\n"
        
        return response
    
    def summarize_news(self, articles: list, topic: str) -> str:
        """
        Create a simple summary of the news
        """
        if not articles:
            return "No articles to summarize."
        
        response = f"📋 News Summary for '{topic}':\n\n"
        
        # Group by source
        by_source = {}
        for article in articles:
            source = article.get('source', 'unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(article)
        
        # Summary for each source
        for source, source_articles in by_source.items():
            response += f"**{source.upper()}** ({len(source_articles)} articles):\n"
            
            # Get key themes (simple keyword extraction)
            all_text = ' '.join([a.get('title', '') + ' ' + a.get('snippet', '') 
                               for a in source_articles[:3]])
            
            response += f"Top stories include:\n"
            for article in source_articles[:2]:
                response += f"• {article['title']}\n"
            response += "\n"
        
        return response
    
    def compare_sources(self, results: dict) -> str:
        """
        Compare coverage across sources
        """
        response = "📈 Source Comparison:\n\n"
        
        for site in ["bbc", "reuters", "techcrunch"]:
            if site in results and isinstance(results[site], list):
                articles = results[site]
                if articles:
                    response += f"**{site.upper()}:**\n"
                    response += f"• Articles found: {len(articles)}\n"
                    response += f"• Top story: {articles[0]['title']}\n\n"
        
        return response
    
    def get_headlines(self, articles: list, topic: str) -> str:
        """
        Get main headlines
        """
        if not articles:
            return "No headlines found."
        
        response = f"📰 Headlines for '{topic}':\n\n"
        
        for i, article in enumerate(articles[:10], 1):
            response += f"{i}. {article['title']}\n"
            response += f"   ({article['source'].upper()})\n"
        
        return response
    
    def chat(self):
        """
        Interactive chat mode
        """
        print("\n🤖 News Chatbot")
        print("=" * 50)
        print("Ask me about any news topic!")
        print("Examples:")
        print("  - What's the latest news about AI?")
        print("  - How many articles about climate change?")
        print("  - Summarize technology news")
        print("  - Which site has more coverage of elections?")
        print("\nType 'quit' to exit, 'clear' to clear cache")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye! 👋")
                    break
                
                elif user_input.lower() == 'clear':
                    self.cached_results.clear()
                    print("Cache cleared!")
                    continue
                
                elif user_input.lower() == 'help':
                    print("\nI can help you with:")
                    print("• Fetching news about any topic")
                    print("• Counting articles from different sources")
                    print("• Summarizing news coverage")
                    print("• Comparing source coverage")
                    print("• Finding latest headlines")
                    continue
                
                elif not user_input:
                    continue
                
                # Get response
                print("\nBot: ", end="")
                response = self.answer_question(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}")


def quick_news(topic: str):
    """
    Quick function that other scripts can use
    
    Example:
        from simple_chatbot import quick_news
        news = quick_news("bitcoin")
    """
    bot = SimpleNewsBot()
    results = bot.fetch_news(topic)
    articles = get_all_articles(results)
    
    summary = {
        "topic": topic,
        "total_articles": len(articles),
        "top_headlines": [a['title'] for a in articles[:5]],
        "sources": {
            "bbc": len(results.get('bbc', [])),
            "reuters": len(results.get('reuters', [])),
            "techcrunch": len(results.get('techcrunch', []))
        }
    }
    
    return summary


def main():
    """
    Main entry point
    """
    bot = SimpleNewsBot()
    
    # Check if running with command line argument
    import sys
    if len(sys.argv) > 1:
        # Direct query mode
        query = ' '.join(sys.argv[1:])
        print(f"\nFetching news about: {query}")
        results = bot.fetch_news(query)
        response = bot.get_headlines(get_all_articles(results), query)
        print(response)
    else:
        # Interactive chat mode
        bot.chat()


if __name__ == "__main__":
    main()
