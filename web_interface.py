"""
Flask Web Interface for SerpAPI Scraper
Provides a modern web UI for searching and managing scraped data
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
from flask_cors import CORS
import os
import json
import sqlite3
from datetime import datetime, timedelta
import threading
from pathlib import Path
import hashlib
import secrets

# Import our modules
from serpapi_scraper import SerpAPIScraper, SmartAnalyzer
from rag_chatbot import RAGChatbot
from monitor_scheduler import TaskScheduler

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

# Global instances
scraper = None
chatbot = None
scheduler = None
analyzer = SmartAnalyzer()

# Initialize components
def init_components():
    global scraper, chatbot, scheduler
    
    api_key = os.getenv("SERPAPI_KEY", "your_api_key_here")
    if api_key == "your_api_key_here":
        return False
    
    scraper = SerpAPIScraper(api_key)
    chatbot = RAGChatbot()
    scheduler = TaskScheduler(api_key)
    scheduler.start()
    
    return True


# HTML Template (inline for simplicity)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SerpAPI Web Scraper</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .card-hover:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .loading {
            display: none;
        }
        .loading.show {
            display: block;
        }
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
        }
        .message {
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
    </style>
</head>
<body class="bg-gray-50">
    <!-- Navigation -->
    <nav class="gradient-bg shadow-lg">
        <div class="container mx-auto px-6 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center">
                    <i class="fas fa-spider text-white text-2xl mr-3"></i>
                    <span class="text-white text-xl font-bold">SerpAPI Scraper</span>
                </div>
                <div class="flex space-x-4">
                    <a href="#search" class="text-white hover:text-gray-200">Search</a>
                    <a href="#chat" class="text-white hover:text-gray-200">Chat</a>
                    <a href="#data" class="text-white hover:text-gray-200">Data</a>
                    <a href="#scheduler" class="text-white hover:text-gray-200">Scheduler</a>
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Container -->
    <div class="container mx-auto px-6 py-8">
        
        <!-- Search Section -->
        <section id="search" class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-6 card-hover">
                <h2 class="text-2xl font-bold mb-6 text-gray-800">
                    <i class="fas fa-search mr-2"></i>Web Search
                </h2>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <label class="block text-gray-700 mb-2">Search Query</label>
                        <input type="text" id="searchQuery" 
                               class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-purple-500"
                               placeholder="Enter your search query...">
                    </div>
                    
                    <div>
                        <label class="block text-gray-700 mb-2">Search Type</label>
                        <select id="searchType" 
                                class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-purple-500">
                            <option value="google">Google Search</option>
                            <option value="news">News Search</option>
                            <option value="scholar">Scholar Search</option>
                            <option value="images">Image Search</option>
                        </select>
                    </div>
                </div>
                
                <div class="mt-6 flex space-x-4">
                    <button onclick="performSearch()" 
                            class="bg-purple-600 text-white px-6 py-2 rounded-lg hover:bg-purple-700 transition">
                        <i class="fas fa-search mr-2"></i>Search
                    </button>
                    <button onclick="performBulkSearch()" 
                            class="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition">
                        <i class="fas fa-layer-group mr-2"></i>Bulk Search
                    </button>
                    <button onclick="enrichContent()" 
                            class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition">
                        <i class="fas fa-file-alt mr-2"></i>Extract Full Content
                    </button>
                </div>
                
                <!-- Loading Indicator -->
                <div id="loadingIndicator" class="loading mt-4">
                    <div class="flex items-center justify-center">
                        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
                        <span class="ml-3 text-gray-600">Searching...</span>
                    </div>
                </div>
                
                <!-- Results Display -->
                <div id="searchResults" class="mt-6"></div>
            </div>
        </section>
        
        <!-- Chat Section -->
        <section id="chat" class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-6 card-hover">
                <h2 class="text-2xl font-bold mb-6 text-gray-800">
                    <i class="fas fa-robot mr-2"></i>RAG Chatbot
                </h2>
                
                <div class="chat-container bg-gray-50 rounded-lg p-4 mb-4" id="chatMessages">
                    <div class="text-center text-gray-500 py-8">
                        <i class="fas fa-comments text-4xl mb-2"></i>
                        <p>Start a conversation about your scraped data!</p>
                    </div>
                </div>
                
                <div class="flex space-x-2">
                    <input type="text" id="chatInput" 
                           class="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:border-purple-500"
                           placeholder="Ask about your data..."
                           onkeypress="if(event.key==='Enter') sendChatMessage()">
                    <button onclick="sendChatMessage()" 
                            class="bg-purple-600 text-white px-6 py-2 rounded-lg hover:bg-purple-700 transition">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
        </section>
        
        <!-- Data Management Section -->
        <section id="data" class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-6 card-hover">
                <h2 class="text-2xl font-bold mb-6 text-gray-800">
                    <i class="fas fa-database mr-2"></i>Data Management
                </h2>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="bg-blue-50 p-4 rounded-lg">
                        <h3 class="font-semibold text-blue-800 mb-2">
                            <i class="fas fa-chart-line mr-2"></i>Statistics
                        </h3>
                        <div id="dataStats" class="text-sm text-gray-700">
                            Loading statistics...
                        </div>
                    </div>
                    
                    <div class="bg-green-50 p-4 rounded-lg">
                        <h3 class="font-semibold text-green-800 mb-2">
                            <i class="fas fa-download mr-2"></i>Export Data
                        </h3>
                        <div class="space-y-2">
                            <button onclick="exportData('json')" 
                                    class="w-full bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 text-sm">
                                Export as JSON
                            </button>
                            <button onclick="exportData('csv')" 
                                    class="w-full bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 text-sm">
                                Export as CSV
                            </button>
                            <button onclick="exportData('markdown')" 
                                    class="w-full bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 text-sm">
                                Export as Markdown
                            </button>
                        </div>
                    </div>
                    
                    <div class="bg-red-50 p-4 rounded-lg">
                        <h3 class="font-semibold text-red-800 mb-2">
                            <i class="fas fa-cog mr-2"></i>Actions
                        </h3>
                        <div class="space-y-2">
                            <button onclick="rebuildIndex()" 
                                    class="w-full bg-yellow-600 text-white px-4 py-2 rounded hover:bg-yellow-700 text-sm">
                                Rebuild Index
                            </button>
                            <button onclick="clearCache()" 
                                    class="w-full bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 text-sm">
                                Clear Cache
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- Recent Data Table -->
                <div class="mt-6">
                    <h3 class="font-semibold text-gray-800 mb-3">Recent Scraped Data</h3>
                    <div id="recentData" class="overflow-x-auto">
                        <p class="text-gray-500">Loading recent data...</p>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Scheduler Section -->
        <section id="scheduler" class="mb-12">
            <div class="bg-white rounded-lg shadow-lg p-6 card-hover">
                <h2 class="text-2xl font-bold mb-6 text-gray-800">
                    <i class="fas fa-clock mr-2"></i>Task Scheduler
                </h2>
                
                <div id="schedulerStatus" class="mb-6">
                    <!-- Scheduler status will be loaded here -->
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <h3 class="font-semibold mb-3">Add New Task</h3>
                        <input type="text" id="taskName" placeholder="Task name" 
                               class="w-full px-3 py-2 border rounded mb-2">
                        <input type="text" id="taskQuery" placeholder="Search query" 
                               class="w-full px-3 py-2 border rounded mb-2">
                        <input type="number" id="taskInterval" placeholder="Interval (minutes)" 
                               class="w-full px-3 py-2 border rounded mb-2">
                        <button onclick="addScheduledTask()" 
                                class="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700">
                            Add Task
                        </button>
                    </div>
                    
                    <div>
                        <h3 class="font-semibold mb-3">Task List</h3>
                        <div id="taskList" class="bg-gray-50 p-3 rounded">
                            Loading tasks...
                        </div>
                    </div>
                </div>
            </div>
        </section>
    </div>

    <!-- JavaScript -->
    <script>
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadDataStats();
            loadRecentData();
            loadSchedulerStatus();
        });
        
        // Search Functions
        async function performSearch() {
            const query = document.getElementById('searchQuery').value;
            const searchType = document.getElementById('searchType').value;
            
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            document.getElementById('loadingIndicator').classList.add('show');
            document.getElementById('searchResults').innerHTML = '';
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query, searchType})
                });
                
                const data = await response.json();
                displayResults(data.results);
            } catch (error) {
                console.error('Search error:', error);
                alert('Search failed: ' + error.message);
            } finally {
                document.getElementById('loadingIndicator').classList.remove('show');
            }
        }
        
        function displayResults(results) {
            const container = document.getElementById('searchResults');
            
            if (!results || results.length === 0) {
                container.innerHTML = '<p class="text-gray-500">No results found</p>';
                return;
            }
            
            const html = results.map(result => `
                <div class="border-b pb-4 mb-4">
                    <h4 class="font-semibold text-lg mb-1">
                        <a href="${result.url}" target="_blank" class="text-blue-600 hover:underline">
                            ${result.title}
                        </a>
                    </h4>
                    <p class="text-sm text-gray-600 mb-2">${result.source} - ${result.date}</p>
                    <p class="text-gray-700">${result.snippet}</p>
                </div>
            `).join('');
            
            container.innerHTML = html;
        }
        
        // Chat Functions
        async function sendChatMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message to chat
            addChatMessage('user', message);
            input.value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message})
                });
                
                const data = await response.json();
                addChatMessage('bot', data.response, data.sources);
            } catch (error) {
                addChatMessage('bot', 'Sorry, I encountered an error. Please try again.');
            }
        }
        
        function addChatMessage(sender, message, sources) {
            const container = document.getElementById('chatMessages');
            
            // Clear placeholder if exists
            if (container.querySelector('.text-center')) {
                container.innerHTML = '';
            }
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message mb-4 ${sender === 'user' ? 'text-right' : 'text-left'}`;
            
            const bubbleClass = sender === 'user' 
                ? 'bg-purple-600 text-white' 
                : 'bg-gray-200 text-gray-800';
            
            let sourcesHtml = '';
            if (sources && sources.length > 0) {
                sourcesHtml = `
                    <div class="mt-2 text-xs text-gray-600">
                        <strong>Sources:</strong>
                        ${sources.map(s => `<a href="${s.url}" target="_blank" class="text-blue-600 hover:underline">${s.title}</a>`).join(', ')}
                    </div>
                `;
            }
            
            messageDiv.innerHTML = `
                <div class="inline-block max-w-2xl">
                    <div class="${bubbleClass} px-4 py-2 rounded-lg">
                        ${message}
                    </div>
                    ${sourcesHtml}
                </div>
            `;
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
        
        // Data Functions
        async function loadDataStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                document.getElementById('dataStats').innerHTML = `
                    <p>Total Documents: <strong>${stats.total_documents}</strong></p>
                    <p>Unique Sources: <strong>${stats.unique_sources}</strong></p>
                    <p>Last Updated: <strong>${stats.last_updated}</strong></p>
                    <p>Index Size: <strong>${stats.index_size} docs</strong></p>
                `;
            } catch (error) {
                console.error('Failed to load stats:', error);
            }
        }
        
        async function loadRecentData() {
            try {
                const response = await fetch('/api/recent');
                const data = await response.json();
                
                if (data.length === 0) {
                    document.getElementById('recentData').innerHTML = '<p class="text-gray-500">No recent data</p>';
                    return;
                }
                
                const html = `
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Title</th>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Source</th>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200">
                            ${data.map(item => `
                                <tr>
                                    <td class="px-4 py-2 text-sm">
                                        <a href="${item.url}" target="_blank" class="text-blue-600 hover:underline">
                                            ${item.title}
                                        </a>
                                    </td>
                                    <td class="px-4 py-2 text-sm text-gray-600">${item.source}</td>
                                    <td class="px-4 py-2 text-sm text-gray-600">${item.scrape_timestamp}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
                
                document.getElementById('recentData').innerHTML = html;
            } catch (error) {
                console.error('Failed to load recent data:', error);
            }
        }
        
        // Export Functions
        async function exportData(format) {
            try {
                const response = await fetch(`/api/export/${format}`);
                const blob = await response.blob();
                
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `scraped_data_${Date.now()}.${format}`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } catch (error) {
                alert('Export failed: ' + error.message);
            }
        }
        
        // Scheduler Functions
        async function loadSchedulerStatus() {
            try {
                const response = await fetch('/api/scheduler/status');
                const status = await response.json();
                
                document.getElementById('schedulerStatus').innerHTML = `
                    <div class="flex items-center justify-between">
                        <div>
                            <span class="text-sm text-gray-600">Scheduler Status:</span>
                            <span class="ml-2 px-3 py-1 rounded-full text-xs font-semibold ${
                                status.running ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                            }">
                                ${status.running ? 'Running' : 'Stopped'}
                            </span>
                        </div>
                        <div class="text-sm text-gray-600">
                            Success Rate (24h): <strong>${status.metrics.success_rate.toFixed(1)}%</strong>
                        </div>
                    </div>
                `;
                
                // Load task list
                const taskHtml = Object.entries(status.tasks).map(([name, task]) => `
                    <div class="mb-2 p-2 bg-white rounded border">
                        <div class="font-semibold">${name}</div>
                        <div class="text-sm text-gray-600">${task.query}</div>
                        <div class="text-xs text-gray-500">Last run: ${task.last_run || 'Never'}</div>
                        <button onclick="runTaskNow('${name}')" class="mt-1 text-xs bg-blue-500 text-white px-2 py-1 rounded">
                            Run Now
                        </button>
                    </div>
                `).join('');
                
                document.getElementById('taskList').innerHTML = taskHtml || '<p class="text-gray-500">No tasks scheduled</p>';
            } catch (error) {
                console.error('Failed to load scheduler status:', error);
            }
        }
        
        async function addScheduledTask() {
            const name = document.getElementById('taskName').value;
            const query = document.getElementById('taskQuery').value;
            const interval = document.getElementById('taskInterval').value;
            
            if (!name || !query || !interval) {
                alert('Please fill all fields');
                return;
            }
            
            try {
                await fetch('/api/scheduler/add', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name, query, interval: parseInt(interval)})
                });
                
                alert('Task added successfully');
                loadSchedulerStatus();
            } catch (error) {
                alert('Failed to add task: ' + error.message);
            }
        }
        
        async function runTaskNow(name) {
            try {
                await fetch(`/api/scheduler/run/${name}`, {method: 'POST'});
                alert(`Task '${name}' started`);
            } catch (error) {
                alert('Failed to run task: ' + error.message);
            }
        }
        
        // Utility Functions
        async function rebuildIndex() {
            if (!confirm('Rebuild the search index? This may take a while.')) return;
            
            try {
                await fetch('/api/rebuild-index', {method: 'POST'});
                alert('Index rebuild started');
                loadDataStats();
            } catch (error) {
                alert('Failed to rebuild index: ' + error.message);
            }
        }
        
        async function clearCache() {
            if (!confirm('Clear all cached data?')) return;
            
            try {
                await fetch('/api/clear-cache', {method: 'POST'});
                alert('Cache cleared');
            } catch (error) {
                alert('Failed to clear cache: ' + error.message);
            }
        }
    </script>
</body>
</html>
"""

# Routes
@app.route('/')
def index():
    if not scraper:
        return "Please set your SERPAPI_KEY environment variable", 500
    return HTML_TEMPLATE


@app.route('/api/search', methods=['POST'])
def api_search():
    data = request.json
    query = data.get('query')
    search_type = data.get('searchType', 'google')
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    try:
        if search_type == 'google':
            results = scraper.search_google(query)
        elif search_type == 'news':
            results = scraper.search_news(query)
        elif search_type == 'scholar':
            results = scraper.search_scholar(query)
        elif search_type == 'images':
            results = scraper.search_images(query)
        else:
            return jsonify({'error': 'Invalid search type'}), 400
        
        # Convert to dict for JSON serialization
        if search_type != 'images':
            results_dict = [
                {
                    'title': r.title,
                    'url': r.url,
                    'snippet': r.snippet,
                    'source': r.source,
                    'date': r.date
                }
                for r in results
            ]
        else:
            results_dict = results
        
        return jsonify({'results': results_dict})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    message = data.get('message')
    
    if not message:
        return jsonify({'error': 'Message required'}), 400
    
    try:
        response_data = chatbot.generate_response(message)
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    conn = sqlite3.connect('scraped_data.db')
    cursor = conn.cursor()
    
    # Get statistics
    cursor.execute("SELECT COUNT(*) FROM scraped_data")
    total_docs = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT source) FROM scraped_data")
    unique_sources = cursor.fetchone()[0]
    
    cursor.execute("SELECT MAX(created_at) FROM scraped_data")
    last_updated = cursor.fetchone()[0]
    
    conn.close()
    
    # Get index size
    index_size = len(chatbot.vector_store.documents) if chatbot else 0
    
    return jsonify({
        'total_documents': total_docs,
        'unique_sources': unique_sources,
        'last_updated': last_updated or 'Never',
        'index_size': index_size
    })


@app.route('/api/recent')
def api_recent():
    conn = sqlite3.connect('scraped_data.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT title, url, source, scrape_timestamp 
        FROM scraped_data 
        ORDER BY created_at DESC 
        LIMIT 10
    """)
    
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    conn.close()
    return jsonify(results)


@app.route('/api/export/<format>')
def api_export(format):
    conn = sqlite3.connect('scraped_data.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM scraped_data ORDER BY created_at DESC LIMIT 100")
    columns = [desc[0] for desc in cursor.description]
    data = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format == 'json':
        filename = f"export_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == 'csv':
        import csv
        filename = f"export_{timestamp}.csv"
        with open(filename, 'w', newline='') as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(data)
    elif format == 'markdown':
        filename = f"export_{timestamp}.md"
        with open(filename, 'w') as f:
            f.write("# Scraped Data Export\n\n")
            for item in data:
                f.write(f"## {item.get('title', 'No Title')}\n")
                f.write(f"**URL:** {item.get('url', '')}\n")
                f.write(f"**Source:** {item.get('source', '')}\n")
                f.write(f"**Snippet:** {item.get('snippet', '')}\n")
                f.write("\n---\n\n")
    else:
        return jsonify({'error': 'Invalid format'}), 400
    
    return send_file(filename, as_attachment=True)


@app.route('/api/scheduler/status')
def api_scheduler_status():
    if not scheduler:
        return jsonify({'error': 'Scheduler not initialized'}), 500
    return jsonify(scheduler.get_status())


@app.route('/api/scheduler/add', methods=['POST'])
def api_scheduler_add():
    data = request.json
    name = data.get('name')
    query = data.get('query')
    interval = data.get('interval', 60)
    
    if not name or not query:
        return jsonify({'error': 'Name and query required'}), 400
    
    scheduler.add_recurring_task(name, query, interval)
    return jsonify({'success': True})


@app.route('/api/scheduler/run/<name>', methods=['POST'])
def api_scheduler_run(name):
    result = scheduler.run_task_now(name)
    if result:
        return jsonify({'success': True, 'status': result.status})
    return jsonify({'error': 'Task not found'}), 404


@app.route('/api/rebuild-index', methods=['POST'])
def api_rebuild_index():
    def rebuild():
        chatbot.build_index_from_db()
    
    thread = threading.Thread(target=rebuild)
    thread.start()
    
    return jsonify({'success': True, 'message': 'Index rebuild started'})


@app.route('/api/clear-cache', methods=['POST'])
def api_clear_cache():
    try:
        import shutil
        cache_dir = Path('cache')
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    if not init_components():
        print("Error: Please set your SERPAPI_KEY environment variable")
        print("export SERPAPI_KEY='your_api_key_here'")
        return
    
    print("=" * 50)
    print("SerpAPI Web Interface")
    print("=" * 50)
    print(f"Starting server at http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == "__main__":
    main()
