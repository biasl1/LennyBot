import logging
import time
import re
from datetime import datetime
from modules.database import get_history_collection, get_reminder_collection, get_pin_collection
import chromadb
from config import Config
from chromadb.config import Settings
from typing import Tuple, Dict, List, Any, Optional

class KnowledgeStore:
    """Knowledge management for LennyBot."""
    
    def __init__(self):
        # Use the same client as the rest of the application
        try:
            # Get direct access to the ChromaDB client
            self.client = chromadb.PersistentClient(
                path=Config.CHROMADB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create knowledge collections
            self.knowledge_collection = self.client.get_or_create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.topics_collection = self.client.get_or_create_collection(
                name="knowledge_topics",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logging.error(f"Error initializing knowledge store: {e}")
            self.knowledge_collection = None
            self.topics_collection = None
    
    def store_knowledge(self, text, topic=None, source="user", chat_id=None):
        """Store knowledge with optional topic categorization."""
        if not self.knowledge_collection:
            return False
            
        try:
            # Generate unique ID
            knowledge_id = f"know-{int(time.time())}-{hash(text) % 10000}"
            
            # Normalize topic if provided
            if topic:
                topic = topic.lower().strip()
            else:
                # Default topic
                topic = "general"
            
            # Store in knowledge collection
            self.knowledge_collection.add(
                documents=[text],
                metadatas=[{
                    "topic": topic,
                    "source": source,
                    "timestamp": datetime.now().isoformat(),
                    "chat_id": str(chat_id) if chat_id else "global"
                }],
                ids=[knowledge_id]
            )
            
            # Add to topic collection
            self._add_to_topic(topic, knowledge_id)
                
            return knowledge_id
        except Exception as e:
            logging.error(f"Error storing knowledge: {e}")
            return None
    
    def _add_to_topic(self, topic, knowledge_id):
        """Add knowledge to a topic."""
        if not self.topics_collection:
            return
            
        topic_id = f"topic-{topic.replace(' ', '_')}"
        
        # Check if topic exists
        existing = self.topics_collection.get(
            ids=[topic_id],
            include=["metadatas"]
        )
        
        if existing and len(existing['ids']) > 0:
            # Update existing topic
            metadata = existing['metadatas'][0]
            entries = metadata.get("entries", "").split(",") if metadata.get("entries") else []
            entries.append(knowledge_id)
            
            self.topics_collection.update(
                ids=[topic_id],
                metadatas=[{**metadata, "entries": ",".join(entries), "count": len(entries)}]
            )
        else:
            # Create new topic
            self.topics_collection.add(
                documents=[f"Topic: {topic}"],
                metadatas=[{
                    "name": topic,
                    "entries": knowledge_id,
                    "count": 1,
                    "created_at": datetime.now().isoformat()
                }],
                ids=[topic_id]
            )
    
    def search_knowledge(self, query, limit=3, topic=None):
        """Search knowledge base for information."""
        if not self.knowledge_collection:
            return []
            
        try:
            # Build where clause if topic specified
            where_clause = {"topic": topic} if topic else None
            
            # Run the search
            results = self.knowledge_collection.query(
                query_texts=[query],
                where=where_clause,
                n_results=limit,
                include=["metadatas", "distances"]
            )
            
            # Format results
            knowledge_results = []
            if results and len(results['ids'][0]) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    knowledge_results.append({
                        'content': doc,
                        'topic': results['metadatas'][0][i].get('topic', 'general'),
                        'relevance': 1 - results['distances'][0][i]
                    })
                    
            return knowledge_results
        except Exception as e:
            logging.error(f"Error searching knowledge: {e}")
            return []
    
    def get_topics(self, limit=10):
        """Get list of available topics."""
        if not self.topics_collection:
            return []
            
        try:
            results = self.topics_collection.get(include=["metadatas"])
            
            topics = []
            if results and len(results['ids']) > 0:
                for i, topic_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    topics.append({
                        'name': metadata.get('name', 'unknown'),
                        'count': metadata.get('count', 0)
                    })
                
                # Sort by count
                topics.sort(key=lambda x: x['count'], reverse=True)
                
            return topics[:limit]
        except Exception as e:
            logging.error(f"Error retrieving topics: {e}")
            return []