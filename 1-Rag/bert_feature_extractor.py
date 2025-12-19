"""
BERT-based Feature Extractor for Legal Text.
Uses 'all-MiniLM-L6-v2' to generate dense vector embeddings that capture semantic meaning.
"""

import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not installed. Using fallback zeros.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class BERTFeatureExtractor:
    """Extracts semantic embeddings using a pre-trained Transformer model"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize with a specific model (default: fast & efficient)"""
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading SentenceTransformer model: {model_name}...")
                self.model = SentenceTransformer(model_name)
                logger.info("✅ Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate 384-dimensional embedding for the input text.
        
        Args:
            text: Case description or summary.
            
        Returns:
            numpy array of shape (384,)
        """
        if not self.model:
            # Return zero vector if model is missing (fallback)
            return np.zeros(384)
            
        try:
            # Encode text (convert to numpy automatically)
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return np.zeros(384)

# Singleton instance for easy import
bert_extractor = BERTFeatureExtractor()
