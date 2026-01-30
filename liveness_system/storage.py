"""
Storage Module for Face Embeddings
Handles saving and loading face embeddings with associated names
"""

import pickle
import os
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from . import config


class EmbeddingStorage:
    """Manages storage and retrieval of face embeddings."""
    
    def __init__(self, storage_path: str = None):
        """Initialize storage with specified path."""
        if storage_path is None:
            storage_path = config.EMBEDDINGS_FILE
        
        self.storage_path = storage_path
        self.embeddings: Dict[str, dict] = {}
        self._load()
    
    def _load(self):
        """Load embeddings from file if exists."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
                print(f"Loaded {len(self.embeddings)} registered faces.")
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                self.embeddings = {}
    
    def _save(self):
        """Save embeddings to file."""
        try:
            with open(self.storage_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            return True
        except Exception as e:
            print(f"Error saving embeddings: {e}")
            return False
    
    def register(self, name: str, encoding: np.ndarray) -> bool:
        """Register a new face with associated name."""
        if name in self.embeddings:
            print(f"Name '{name}' already exists. Updating...")
        
        self.embeddings[name] = {
            'encoding': encoding,
            'registered_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        return self._save()
    
    def get_encoding(self, name: str) -> Optional[np.ndarray]:
        """Get encoding for a registered name."""
        if name in self.embeddings:
            return self.embeddings[name]['encoding']
        return None
    
    def get_all_encodings(self) -> Dict[str, np.ndarray]:
        """Get all registered encodings."""
        return {name: data['encoding'] for name, data in self.embeddings.items()}
    
    def get_all_names(self) -> List[str]:
        """Get list of all registered names."""
        return list(self.embeddings.keys())
    
    def exists(self, name: str) -> bool:
        """Check if name is already registered."""
        return name in self.embeddings
    
    def delete(self, name: str) -> bool:
        """Delete a registered face."""
        if name in self.embeddings:
            del self.embeddings[name]
            return self._save()
        return False
    
    def find_match(self, encoding: np.ndarray, tolerance: float = 0.6) -> Optional[str]:
        """Find a matching face in registered embeddings."""
        import face_recognition
        
        if len(self.embeddings) == 0:
            return None
        
        names = list(self.embeddings.keys())
        known_encodings = [self.embeddings[name]['encoding'] for name in names]
        
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance)
        distances = face_recognition.face_distance(known_encodings, encoding)
        
        if True in matches:
            best_match_idx = np.argmin(distances)
            if matches[best_match_idx]:
                return names[best_match_idx]
        
        return None
    
    def count(self) -> int:
        """Get number of registered faces."""
        return len(self.embeddings)
