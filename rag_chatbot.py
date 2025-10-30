"""
Enhanced RAG Chatbot System
Uses scraped data from SerpAPI for intelligent, context-aware responses
"""

import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle

# For embeddings and similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# For text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re

# For chatbot interface
import streamlit as st
from dataclasses import dataclass
import hashlib

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document class for RAG system"""
    id: str
    content: str
    title: str
    source: str
    url: str
    date: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    
    def __hash__(self):
        return hash(self.id)


class VectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        self.id_to_index = {}
        
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents with their embeddings to the store"""
        start_idx = len(self.documents)
        
        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]
            self.documents.append(doc)
            self.id_to_index[doc.id] = start_idx + i
        
        self.index.add(embeddings)
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                similarity = 1 / (1 + distances[0][i])  # Convert distance to similarity
                results.append((doc, similarity))
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk"""
        faiss.write_index(self.index, f"{path}/faiss.index")
        with open(f"{path}/documents.pkl", 'wb') as f:
            pickle.dump((self.documents, self.id_to_index), f)
    
    def load(self, path: str):
        """Load vector store from disk"""
        self.index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/documents.pkl", 'rb') as f:
            self.documents, self.id_to_index = pickle.load(f)


class TextProcessor:
    """Advanced text processing for RAG system"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep overlap
                overlap_size = 0
                overlap_chunk = []
                for sent in reversed(current_chunk):
                    overlap_size += len(sent.split())
                    overlap_chunk.insert(0, sent)
                    if overlap_size >= overlap:
                        break
                
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords using TF-IDF approach"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Simple keyword extraction
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum() and w not in self.stop_words and len(w) > 2]
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:num_keywords]]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """Simple extractive summarization"""
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return text
        
        # Simple scoring based on word frequency
        word_freq = {}
        words = word_tokenize(text.lower())
        for word in words:
            if word.isalnum() and word not in self.stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences
        sentence_scores = {}
        for sentence in sentences:
            words_in_sentence = word_tokenize(sentence.lower())
            score = 0
            for word in words_in_sentence:
                if word in word_freq:
                    score += word_freq[word]
            sentence_scores[sentence] = score / len(words_in_sentence)
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        summary_sentences = [sent for sent, score in top_sentences[:3]]
        
        # Maintain original order
        summary = []
        for sentence in sentences:
            if sentence in summary_sentences:
                summary.append(sentence)
                if len(' '.join(summary)) > max_length:
                    break
        
        return ' '.join(summary)


class RAGChatbot:
    """Main RAG Chatbot class"""
    
    def __init__(self, db_path: str = "scraped_data.db", 
                 vector_store_path: str = "vector_store"):
        self.db_path = db_path
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = VectorStore(embedding_dim=384)
        self.text_processor = TextProcessor()
        
        # Load existing data
        self.load_or_build_index()
        
        # Conversation history
        self.conversation_history = []
    
    def load_or_build_index(self):
        """Load existing index or build new one from database"""
        index_file = self.vector_store_path / "faiss.index"
        
        if index_file.exists():
            logger.info("Loading existing vector index...")
            self.vector_store.load(str(self.vector_store_path))
        else:
            logger.info("Building new vector index from database...")
            self.build_index_from_db()
    
    def build_index_from_db(self):
        """Build vector index from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, url, snippet, source, date, full_content, metadata 
            FROM scraped_data 
            WHERE snippet IS NOT NULL OR full_content IS NOT NULL
        ''')
        
        documents = []
        for row in cursor.fetchall():
            title, url, snippet, source, date, full_content, metadata = row
            
            # Use full content if available, otherwise snippet
            content = full_content if full_content else snippet
            if not content:
                continue
            
            # Clean and process content
            content = self.text_processor.clean_text(content)
            
            # Create chunks for long content
            if len(content.split()) > 500:
                chunks = self.text_processor.chunk_text(content)
            else:
                chunks = [content]
            
            # Create document for each chunk
            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{url}_{i}".encode()).hexdigest()
                doc = Document(
                    id=doc_id,
                    content=chunk,
                    title=title or "",
                    source=source or "",
                    url=url or "",
                    date=date or "",
                    metadata=json.loads(metadata) if metadata else {}
                )
                documents.append(doc)
        
        conn.close()
        
        if documents:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} documents...")
            contents = [doc.content for doc in documents]
            embeddings = self.model.encode(contents, show_progress_bar=True)
            
            # Add to vector store
            self.vector_store.add_documents(documents, embeddings)
            
            # Save index
            self.vector_store.save(str(self.vector_store_path))
            logger.info("Vector index built and saved")
        else:
            logger.warning("No documents found in database")
    
    def search_context(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant context"""
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search vector store
        results = self.vector_store.search(query_embedding[0], k=k)
        
        return [doc for doc, score in results]
    
    def generate_response(self, query: str, use_context: bool = True) -> Dict:
        """Generate response to user query"""
        response = {
            "query": query,
            "response": "",
            "sources": [],
            "context_used": False,
            "timestamp": datetime.now().isoformat()
        }
        
        if use_context:
            # Search for relevant context
            relevant_docs = self.search_context(query, k=5)
            
            if relevant_docs:
                response["context_used"] = True
                
                # Build context string
                context_parts = []
                seen_urls = set()
                
                for doc in relevant_docs:
                    if doc.url not in seen_urls:
                        context_parts.append(f"Source: {doc.source}\nTitle: {doc.title}\n{doc.content}")
                        seen_urls.add(doc.url)
                        response["sources"].append({
                            "title": doc.title,
                            "url": doc.url,
                            "source": doc.source,
                            "date": doc.date
                        })
                
                context = "\n\n---\n\n".join(context_parts[:3])  # Limit to 3 sources
                
                # Generate response based on context
                response["response"] = self._generate_contextual_response(query, context)
            else:
                response["response"] = self._generate_fallback_response(query)
        else:
            response["response"] = self._generate_fallback_response(query)
        
        # Add to conversation history
        self.conversation_history.append(response)
        
        return response
    
    def _generate_contextual_response(self, query: str, context: str) -> str:
        """Generate response using context"""
        # Extract key information from context
        keywords = self.text_processor.extract_keywords(context, num_keywords=10)
        
        # Create a structured response
        response_parts = []
        
        # Add introduction
        response_parts.append(f"Based on the available information about '{query}':\n")
        
        # Summarize context
        summary = self.text_processor.summarize(context, max_length=300)
        response_parts.append(summary)
        
        # Add key points if available
        if keywords:
            response_parts.append(f"\n\nKey topics: {', '.join(keywords[:5])}")
        
        # Add note about sources
        response_parts.append("\n\nThis information is compiled from recent search results. Please check the sources for more detailed information.")
        
        return "\n".join(response_parts)
    
    def _generate_fallback_response(self, query: str) -> str:
        """Generate fallback response when no context is found"""
        return f"I don't have specific information about '{query}' in my current knowledge base. You might want to search for more recent information or rephrase your query."
    
    def update_index(self, new_data: List[Dict]):
        """Update vector index with new data"""
        documents = []
        
        for item in new_data:
            content = item.get('full_content') or item.get('snippet')
            if not content:
                continue
            
            # Clean and chunk content
            content = self.text_processor.clean_text(content)
            chunks = self.text_processor.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{item.get('url', '')}_{i}".encode()).hexdigest()
                doc = Document(
                    id=doc_id,
                    content=chunk,
                    title=item.get('title', ''),
                    source=item.get('source', ''),
                    url=item.get('url', ''),
                    date=item.get('date', ''),
                    metadata=item.get('metadata', {})
                )
                documents.append(doc)
        
        if documents:
            # Generate embeddings and add to store
            contents = [doc.content for doc in documents]
            embeddings = self.model.encode(contents)
            self.vector_store.add_documents(documents, embeddings)
            
            # Save updated index
            self.vector_store.save(str(self.vector_store_path))
            logger.info(f"Updated index with {len(documents)} new documents")
    
    def get_conversation_summary(self) -> str:
        """Get summary of conversation history"""
        if not self.conversation_history:
            return "No conversation history yet."
        
        summary_parts = []
        for i, turn in enumerate(self.conversation_history[-5:], 1):  # Last 5 turns
            summary_parts.append(f"{i}. Q: {turn['query'][:50]}...")
            if turn['context_used']:
                summary_parts.append(f"   A: [Context-based response with {len(turn['sources'])} sources]")
            else:
                summary_parts.append(f"   A: [Direct response]")
        
        return "\n".join(summary_parts)


def create_streamlit_app():
    """Create Streamlit web interface for the chatbot"""
    
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Chatbot with Web Search")
    st.markdown("Ask questions and get answers based on scraped web data!")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        use_context = st.checkbox("Use Context", value=True)
        num_sources = st.slider("Number of Sources", 1, 10, 5)
        
        st.header("📊 Statistics")
        if st.session_state.chatbot.vector_store.documents:
            st.metric("Documents in Index", len(st.session_state.chatbot.vector_store.documents))
        
        if st.button("🔄 Rebuild Index"):
            with st.spinner("Rebuilding index..."):
                st.session_state.chatbot.build_index_from_db()
            st.success("Index rebuilt!")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.markdown(f"- [{source['title']}]({source['url']}) - {source['source']}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = st.session_state.chatbot.generate_response(prompt, use_context=use_context)
            
            st.markdown(response_data["response"])
            
            # Show sources
            if response_data["sources"]:
                with st.expander("📚 Sources"):
                    for source in response_data["sources"]:
                        st.markdown(f"- [{source['title']}]({source['url']}) - {source['source']}")
            
            # Add to messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_data["response"],
                "sources": response_data["sources"]
            })


def main():
    """Main function to run the chatbot"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        # Run Streamlit app
        create_streamlit_app()
    else:
        # Run in CLI mode
        print("🤖 RAG Chatbot - CLI Mode")
        print("Type 'quit' to exit, 'help' for commands")
        print("-" * 50)
        
        chatbot = RAGChatbot()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  quit - Exit the chatbot")
                    print("  help - Show this help")
                    print("  history - Show conversation history")
                    print("  rebuild - Rebuild the vector index")
                    print("  stats - Show statistics")
                
                elif user_input.lower() == 'history':
                    print("\nConversation History:")
                    print(chatbot.get_conversation_summary())
                
                elif user_input.lower() == 'rebuild':
                    print("Rebuilding index...")
                    chatbot.build_index_from_db()
                    print("Index rebuilt!")
                
                elif user_input.lower() == 'stats':
                    print(f"\nStatistics:")
                    print(f"  Documents in index: {len(chatbot.vector_store.documents)}")
                    print(f"  Conversation turns: {len(chatbot.conversation_history)}")
                
                elif user_input:
                    response_data = chatbot.generate_response(user_input)
                    
                    print(f"\nBot: {response_data['response']}")
                    
                    if response_data['sources']:
                        print("\nSources:")
                        for i, source in enumerate(response_data['sources'], 1):
                            print(f"  {i}. {source['title']} - {source['source']}")
                            print(f"     {source['url']}")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
