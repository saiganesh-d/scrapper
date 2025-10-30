"""
Monitoring and Scheduler for Automated Web Scraping
Handles scheduled tasks, monitoring, and notifications
"""

import os
import time
import json
import schedule
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from dataclasses import dataclass, asdict
import sqlite3
from pathlib import Path

# Import our scraper
from serpapi_scraper import SerpAPIScraper, ScrapedData
from rag_chatbot import RAGChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a scheduled task"""
    task_name: str
    status: str  # success, failed, partial
    start_time: datetime
    end_time: datetime
    results_count: int
    error_message: Optional[str] = None
    exported_files: List[str] = None
    
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


class NotificationManager:
    """Handles notifications via email, Slack, Discord, etc."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.email_config = config.get('email', {})
        self.slack_config = config.get('slack', {})
        self.discord_config = config.get('discord', {})
    
    def send_email(self, subject: str, body: str, recipients: List[str] = None):
        """Send email notification"""
        if not self.email_config.get('enabled'):
            return
        
        try:
            recipients = recipients or self.email_config.get('recipients', [])
            if not recipients:
                return
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
            
            logger.info(f"Email sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def send_slack(self, message: str):
        """Send Slack notification"""
        if not self.slack_config.get('enabled'):
            return
        
        try:
            webhook_url = self.slack_config.get('webhook_url')
            if not webhook_url:
                return
            
            payload = {
                'text': message,
                'channel': self.slack_config.get('channel', '#general')
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"Slack notification sent")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def send_discord(self, message: str):
        """Send Discord notification"""
        if not self.discord_config.get('enabled'):
            return
        
        try:
            webhook_url = self.discord_config.get('webhook_url')
            if not webhook_url:
                return
            
            payload = {'content': message}
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"Discord notification sent")
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
    
    def send_notification(self, title: str, message: str, priority: str = "normal"):
        """Send notification to all enabled channels"""
        # Format message
        formatted_message = f"**{title}**\n\n{message}\n\nPriority: {priority}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send to all channels
        self.send_email(title, formatted_message)
        self.send_slack(formatted_message)
        self.send_discord(formatted_message)


class MetricsCollector:
    """Collects and stores performance metrics"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize metrics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS task_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT,
                status TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_seconds REAL,
                results_count INTEGER,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cpu_percent REAL,
                memory_percent REAL,
                disk_usage_percent REAL,
                active_tasks INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_task_result(self, result: TaskResult):
        """Record task execution result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO task_metrics 
            (task_name, status, start_time, end_time, duration_seconds, results_count, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.task_name,
            result.status,
            result.start_time.isoformat(),
            result.end_time.isoformat(),
            result.duration_seconds(),
            result.results_count,
            result.error_message
        ))
        
        conn.commit()
        conn.close()
    
    def record_system_metrics(self):
        """Record system performance metrics"""
        try:
            import psutil
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_metrics 
                (cpu_percent, memory_percent, disk_usage_percent, active_tasks)
                VALUES (?, ?, ?, ?)
            ''', (
                psutil.cpu_percent(interval=1),
                psutil.virtual_memory().percent,
                psutil.disk_usage('/').percent,
                threading.active_count() - 1  # Exclude main thread
            ))
            
            conn.commit()
            conn.close()
        except ImportError:
            logger.warning("psutil not installed - skipping system metrics")
    
    def get_task_stats(self, hours: int = 24) -> Dict:
        """Get task statistics for the last N hours"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since = datetime.now() - timedelta(hours=hours)
        
        # Get success rate
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                AVG(duration_seconds) as avg_duration,
                SUM(results_count) as total_results
            FROM task_metrics
            WHERE created_at > ?
        ''', (since.isoformat(),))
        
        row = cursor.fetchone()
        stats = {
            'total_tasks': row[0],
            'successful': row[1],
            'failed': row[2],
            'avg_duration': row[3],
            'total_results': row[4],
            'success_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0
        }
        
        conn.close()
        return stats


class ScheduledTask:
    """Represents a scheduled scraping task"""
    
    def __init__(self, name: str, query: str, search_type: str = "google",
                 scraper: SerpAPIScraper = None, metrics: MetricsCollector = None,
                 notifier: NotificationManager = None, auto_export: bool = True):
        self.name = name
        self.query = query
        self.search_type = search_type
        self.scraper = scraper
        self.metrics = metrics
        self.notifier = notifier
        self.auto_export = auto_export
        self.last_run = None
    
    def execute(self) -> TaskResult:
        """Execute the scheduled task"""
        logger.info(f"Executing task: {self.name}")
        start_time = datetime.now()
        
        result = TaskResult(
            task_name=self.name,
            status="success",
            start_time=start_time,
            end_time=start_time,  # Will be updated
            results_count=0
        )
        
        try:
            # Perform the search
            if self.search_type == "google":
                data = self.scraper.search_google(self.query)
            elif self.search_type == "news":
                data = self.scraper.search_news(self.query)
            elif self.search_type == "scholar":
                data = self.scraper.search_scholar(self.query)
            else:
                raise ValueError(f"Unknown search type: {self.search_type}")
            
            result.results_count = len(data)
            
            # Auto-export if enabled
            if self.auto_export and data:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                export_dir = Path("exports") / self.name
                export_dir.mkdir(parents=True, exist_ok=True)
                
                exported_files = []
                
                # Export to multiple formats
                json_file = export_dir / f"{timestamp}.json"
                self.scraper.export_to_json(data, str(json_file))
                exported_files.append(str(json_file))
                
                csv_file = export_dir / f"{timestamp}.csv"
                self.scraper.export_to_csv(data, str(csv_file))
                exported_files.append(str(csv_file))
                
                result.exported_files = exported_files
                logger.info(f"Exported results to: {export_dir}")
            
            # Update RAG index if we have full content
            if hasattr(self, 'chatbot') and self.chatbot:
                enriched_data = self.scraper.enrich_with_full_content(data[:3])
                new_data = [asdict(item) for item in enriched_data if item.full_content]
                if new_data:
                    self.chatbot.update_index(new_data)
            
            self.last_run = datetime.now()
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            logger.error(f"Task {self.name} failed: {e}")
            
            # Send alert for failed tasks
            if self.notifier:
                self.notifier.send_notification(
                    f"Task Failed: {self.name}",
                    f"Error: {result.error_message}",
                    priority="high"
                )
        
        finally:
            result.end_time = datetime.now()
            
            # Record metrics
            if self.metrics:
                self.metrics.record_task_result(result)
            
            # Send success notification for important tasks
            if result.status == "success" and result.results_count > 0 and self.notifier:
                self.notifier.send_notification(
                    f"Task Completed: {self.name}",
                    f"Found {result.results_count} results\nDuration: {result.duration_seconds():.2f}s",
                    priority="normal"
                )
        
        return result


class TaskScheduler:
    """Main scheduler for running tasks"""
    
    def __init__(self, api_key: str, config: Dict = None):
        self.api_key = api_key
        self.config = config or {}
        
        # Initialize components
        self.scraper = SerpAPIScraper(api_key)
        self.metrics = MetricsCollector()
        self.notifier = NotificationManager(config.get('notifications', {}))
        self.chatbot = RAGChatbot() if config.get('enable_rag', True) else None
        
        # Task registry
        self.tasks = {}
        
        # Threading
        self.scheduler_thread = None
        self.running = False
    
    def add_task(self, name: str, query: str, schedule_str: str, 
                 search_type: str = "google", auto_export: bool = True):
        """Add a scheduled task"""
        task = ScheduledTask(
            name=name,
            query=query,
            search_type=search_type,
            scraper=self.scraper,
            metrics=self.metrics,
            notifier=self.notifier,
            auto_export=auto_export
        )
        
        if hasattr(self, 'chatbot'):
            task.chatbot = self.chatbot
        
        self.tasks[name] = task
        
        # Schedule the task
        schedule.every().day.at(schedule_str).do(task.execute)
        logger.info(f"Scheduled task '{name}' at {schedule_str}")
    
    def add_recurring_task(self, name: str, query: str, interval_minutes: int,
                          search_type: str = "google", auto_export: bool = True):
        """Add a recurring task with minute interval"""
        task = ScheduledTask(
            name=name,
            query=query,
            search_type=search_type,
            scraper=self.scraper,
            metrics=self.metrics,
            notifier=self.notifier,
            auto_export=auto_export
        )
        
        if hasattr(self, 'chatbot'):
            task.chatbot = self.chatbot
        
        self.tasks[name] = task
        
        # Schedule the recurring task
        schedule.every(interval_minutes).minutes.do(task.execute)
        logger.info(f"Scheduled recurring task '{name}' every {interval_minutes} minutes")
    
    def run_task_now(self, name: str) -> Optional[TaskResult]:
        """Run a specific task immediately"""
        if name in self.tasks:
            return self.tasks[name].execute()
        else:
            logger.error(f"Task '{name}' not found")
            return None
    
    def start(self):
        """Start the scheduler in a background thread"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                schedule.run_pending()
                
                # Record system metrics every 5 minutes
                if datetime.now().minute % 5 == 0:
                    self.metrics.record_system_metrics()
                
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
    
    def get_status(self) -> Dict:
        """Get scheduler status"""
        return {
            'running': self.running,
            'tasks': {
                name: {
                    'query': task.query,
                    'search_type': task.search_type,
                    'last_run': task.last_run.isoformat() if task.last_run else None,
                    'auto_export': task.auto_export
                }
                for name, task in self.tasks.items()
            },
            'metrics': self.metrics.get_task_stats(24)
        }


def main():
    """Main function to run the monitoring system"""
    
    # Load configuration
    try:
        from config import (
            SERPAPI_KEY, SCHEDULED_TASKS, NOTIFICATION_CONFIG,
            MONITORING_CONFIG, RAG_CONFIG
        )
    except ImportError:
        print("Please create a config.py file from config_template.py")
        return
    
    # Initialize scheduler
    config = {
        'notifications': NOTIFICATION_CONFIG,
        'monitoring': MONITORING_CONFIG,
        'enable_rag': RAG_CONFIG.get('enabled', True)
    }
    
    scheduler = TaskScheduler(SERPAPI_KEY, config)
    
    # Add scheduled tasks from config
    for task_config in SCHEDULED_TASKS:
        if 'schedule' in task_config:
            scheduler.add_task(
                name=task_config['name'],
                query=task_config['query'],
                schedule_str=task_config['schedule'],
                search_type=task_config.get('search_type', 'google'),
                auto_export=task_config.get('auto_export', True)
            )
        elif 'interval_minutes' in task_config:
            scheduler.add_recurring_task(
                name=task_config['name'],
                query=task_config['query'],
                interval_minutes=task_config['interval_minutes'],
                search_type=task_config.get('search_type', 'google'),
                auto_export=task_config.get('auto_export', True)
            )
    
    # Start scheduler
    scheduler.start()
    
    print("=" * 50)
    print("Web Scraping Monitor & Scheduler")
    print("=" * 50)
    print("\nCommands:")
    print("  status - Show scheduler status")
    print("  run <task_name> - Run a task immediately")
    print("  list - List all tasks")
    print("  metrics - Show performance metrics")
    print("  test-notify - Test notifications")
    print("  quit - Exit")
    print("-" * 50)
    
    try:
        while True:
            command = input("\n> ").strip().lower()
            
            if command == 'quit':
                print("Shutting down...")
                scheduler.stop()
                break
            
            elif command == 'status':
                status = scheduler.get_status()
                print(f"\nScheduler: {'Running' if status['running'] else 'Stopped'}")
                print(f"Active tasks: {len(status['tasks'])}")
                print(f"Success rate (24h): {status['metrics']['success_rate']:.1f}%")
            
            elif command.startswith('run '):
                task_name = command[4:]
                print(f"Running task '{task_name}'...")
                result = scheduler.run_task_now(task_name)
                if result:
                    print(f"Status: {result.status}")
                    print(f"Results: {result.results_count}")
                    print(f"Duration: {result.duration_seconds():.2f}s")
            
            elif command == 'list':
                print("\nScheduled Tasks:")
                for name, info in scheduler.get_status()['tasks'].items():
                    print(f"  - {name}: {info['query']} ({info['search_type']})")
                    if info['last_run']:
                        print(f"    Last run: {info['last_run']}")
            
            elif command == 'metrics':
                metrics = scheduler.metrics.get_task_stats(24)
                print("\nPerformance Metrics (24h):")
                print(f"  Total tasks: {metrics['total_tasks']}")
                print(f"  Successful: {metrics['successful']}")
                print(f"  Failed: {metrics['failed']}")
                print(f"  Success rate: {metrics['success_rate']:.1f}%")
                print(f"  Avg duration: {metrics['avg_duration']:.2f}s")
                print(f"  Total results: {metrics['total_results']}")
            
            elif command == 'test-notify':
                print("Sending test notification...")
                scheduler.notifier.send_notification(
                    "Test Notification",
                    "This is a test notification from the monitoring system",
                    priority="low"
                )
                print("Notification sent!")
            
            else:
                print("Unknown command. Type 'quit' to exit.")
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        scheduler.stop()


if __name__ == "__main__":
    main()
