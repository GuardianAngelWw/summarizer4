import os
import json
import logging
from typing import List, Dict, Optional
from google.cloud import firestore
from google.cloud.exceptions import NotFound

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firestore client
db = firestore.Client()

# Collection names
ENTRIES_COLLECTION = "entries"
SETTINGS_COLLECTION = "settings"

def read_entries(category: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Read entries from Firestore with optional filtering by category.
    Falls back to CSV if Firestore is not available.
    """
    try:
        entries = []
        # Create query
        query = db.collection(ENTRIES_COLLECTION)
        if category:
            query = query.where("category", "==", category)
        
        # Execute query
        docs = query.stream()
        for doc in docs:
            entry = doc.to_dict()
            # Ensure consistent format
            if "category" not in entry:
                entry["category"] = "General"
            entries.append(entry)
        
        logger.info(f"Retrieved {len(entries)} entries from Firestore")
        return entries
    except Exception as e:
        logger.error(f"Firestore error in read_entries: {e}")
        # Fall back to CSV file if available
        try:
            from ollama_telegram_bot import read_entries as csv_read_entries
            logger.warning("Falling back to CSV storage")
            return csv_read_entries(category)
        except Exception as fallback_error:
            logger.error(f"Fallback to CSV also failed: {fallback_error}")
            return []

def write_entries(entries: List[Dict[str, str]]) -> bool:
    """
    Write entries to Firestore.
    Falls back to CSV if Firestore is not available.
    """
    try:
        batch = db.batch()
        
        # First, delete existing entries
        existing = db.collection(ENTRIES_COLLECTION).stream()
        for doc in existing:
            batch.delete(doc.reference)
        
        # Then add new entries
        for i, entry in enumerate(entries):
            # Ensure category exists
            if "category" not in entry or not entry["category"]:
                entry["category"] = "General"
                
            doc_ref = db.collection(ENTRIES_COLLECTION).document(f"entry_{i}")
            batch.set(doc_ref, entry)
        
        # Commit the batch
        batch.commit()
        logger.info(f"Successfully wrote {len(entries)} entries to Firestore")
        return True
    except Exception as e:
        logger.error(f"Firestore error in write_entries: {e}")
        # Fall back to CSV file
        try:
            from ollama_telegram_bot import write_entries as csv_write_entries
            logger.warning("Falling back to CSV storage")
            return csv_write_entries(entries)
        except Exception as fallback_error:
            logger.error(f"Fallback to CSV also failed: {fallback_error}")
            return False

def add_entry(text: str, link: str, category: str = "General") -> bool:
    """
    Add a new entry to Firestore.
    """
    try:
        # Check for duplicates
        entries = read_entries()
        for entry in entries:
            if entry.get("text") == text and entry.get("link") == link:
                logger.info("Entry already exists")
                return False
        
        # Add category if needed
        add_category(category)
        
        # Add new entry
        new_entry = {
            "text": text,
            "link": link,
            "category": category
        }
        
        # Get next index
        next_idx = len(entries)
        db.collection(ENTRIES_COLLECTION).document(f"entry_{next_idx}").set(new_entry)
        logger.info(f"Added new entry with text: {text[:30]}...")
        return True
    except Exception as e:
        logger.error(f"Firestore error in add_entry: {e}")
        # Fall back to CSV file
        try:
            from ollama_telegram_bot import add_entry as csv_add_entry
            logger.warning("Falling back to CSV storage")
            return csv_add_entry(text, link, category)
        except Exception as fallback_error:
            logger.error(f"Fallback to CSV also failed: {fallback_error}")
            return False

def delete_entry(index: int) -> bool:
    """
    Delete an entry by index from Firestore.
    """
    try:
        entries = read_entries()
        if 0 <= index < len(entries):
            # Delete the document
            db.collection(ENTRIES_COLLECTION).document(f"entry_{index}").delete()
            
            # Reindex remaining entries
            remaining_entries = [entries[i] for i in range(len(entries)) if i != index]
            return write_entries(remaining_entries)
        return False
    except Exception as e:
        logger.error(f"Firestore error in delete_entry: {e}")
        # Fall back to CSV file
        try:
            from ollama_telegram_bot import delete_entry as csv_delete_entry
            logger.warning("Falling back to CSV storage")
            return csv_delete_entry(index)
        except Exception as fallback_error:
            logger.error(f"Fallback to CSV also failed: {fallback_error}")
            return False

def get_categories() -> List[str]:
    """
    Get list of categories from Firestore.
    """
    try:
        doc_ref = db.collection(SETTINGS_COLLECTION).document("categories")
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            return data.get("categories", ["General"])
        else:
            # Initialize with default categories
            default_categories = ["General", "Documentation", "Tutorials", "References"]
            doc_ref.set({"categories": default_categories})
            return default_categories
    except Exception as e:
        logger.error(f"Firestore error in get_categories: {e}")
        # Fall back to CSV file
        try:
            from ollama_telegram_bot import get_categories as csv_get_categories
            logger.warning("Falling back to CSV storage")
            return csv_get_categories()
        except Exception as fallback_error:
            logger.error(f"Fallback to CSV also failed: {fallback_error}")
            return ["General"]

def add_category(category: str) -> bool:
    """
    Add a new category to Firestore if it doesn't exist.
    """
    if not category:
        return False
        
    try:
        categories = get_categories()
        if category in categories:
            return True
            
        categories.append(category)
        db.collection(SETTINGS_COLLECTION).document("categories").set({"categories": categories})
        logger.info(f"Added new category: {category}")
        return True
    except Exception as e:
        logger.error(f"Firestore error in add_category: {e}")
        # Fall back to CSV file
        try:
            from ollama_telegram_bot import add_category as csv_add_category
            logger.warning("Falling back to CSV storage")
            return csv_add_category(category)
        except Exception as fallback_error:
            logger.error(f"Fallback to CSV also failed: {fallback_error}")
            return False

def clear_all_entries(category: Optional[str] = None) -> int:
    """
    Clear all entries, optionally filtered by category.
    Returns the number of entries deleted.
    """
    try:
        batch = db.batch()
        deleted_count = 0
        
        if category:
            # Delete only entries in the specified category
            query = db.collection(ENTRIES_COLLECTION).where("category", "==", category)
            docs = query.stream()
            for doc in docs:
                batch.delete(doc.reference)
                deleted_count += 1
                
            # If we deleted entries, we need to reindex the remaining ones
            if deleted_count > 0:
                entries = read_entries()
                filtered_entries = [entry for entry in entries if entry.get("category") != category]
                write_entries(filtered_entries)
        else:
            # Delete all entries
            docs = db.collection(ENTRIES_COLLECTION).stream()
            for doc in docs:
                batch.delete(doc.reference)
                deleted_count += 1
        
        # Commit the batch
        batch.commit()
        logger.info(f"Cleared {deleted_count} entries from Firestore")
        return deleted_count
    except Exception as e:
        logger.error(f"Firestore error in clear_all_entries: {e}")
        # Fall back to CSV file
        try:
            from ollama_telegram_bot import clear_all_entries as csv_clear_all_entries
            logger.warning("Falling back to CSV storage")
            return csv_clear_all_entries(category)
        except Exception as fallback_error:
            logger.error(f"Fallback to CSV also failed: {fallback_error}")
            return 0

def search_entries(query: str, category: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Basic search for entries matching query.
    Note: This is a simple implementation. For better search, consider using Firestore's
    full-text search capabilities or integrating with Algolia or Elasticsearch.
    """
    entries = read_entries(category)
    if not query:
        return entries
    
    query = query.lower()
    return [entry for entry in entries if
            query in entry.get("text", "").lower() or
            query in entry.get("category", "").lower()]

# Helper function to migrate data from CSV to Firestore
def migrate_from_csv():
    """
    Migrate data from CSV files to Firestore.
    """
    try:
        from ollama_telegram_bot import read_entries as csv_read_entries
        from ollama_telegram_bot import get_categories as csv_get_categories
        
        # Migrate entries
        entries = csv_read_entries()
        if entries:
            write_entries(entries)
            logger.info(f"Migrated {len(entries)} entries from CSV to Firestore")
        
        # Migrate categories
        categories = csv_get_categories()
        if categories:
            db.collection(SETTINGS_COLLECTION).document("categories").set({"categories": categories})
            logger.info(f"Migrated {len(categories)} categories from CSV to Firestore")
        
        return True
    except Exception as e:
        logger.error(f"Error migrating from CSV to Firestore: {e}")
        return False 