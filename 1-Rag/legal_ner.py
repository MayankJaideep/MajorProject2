"""
Named Entity Recognition (NER) for Legal Documents
Extract judges, courts, parties, dates, statutes, and other legal entities from text.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
from typing import List, Dict, Optional
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class LegalNER:
    """
    Extract legal entities from text using NER models
    """
    
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """
        Initialize NER model
        
        Args:
            model_name: HuggingFace model for NER
        """
        print(f"🔄 Loading NER model: {model_name}")
        
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            self.ner = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple",
                device=device
            )
            print(f"✅ NER model loaded on {'GPU' if device == 0 else 'CPU'}")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("   Falling back to default NER model...")
            self.ner = pipeline("ner", aggregation_strategy="simple", device=device)
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract all entities from text
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with entities grouped by type
        """
        if not text or len(text.strip()) < 10:
            return {'error': 'Text too short for entity extraction'}
        
        try:
            # Run NER
            entities = self.ner(text[:512])  # Limit to 512 tokens
            
            # Group by entity type
            grouped = defaultdict(list)
            
            for entity in entities:
                entity_type = entity['entity_group']
                entity_text = entity['word']
                confidence = entity['score']
                
                grouped[entity_type].append({
                    'text': entity_text,
                    'confidence': round(confidence, 3),
                    'start': entity.get('start', 0),
                    'end': entity.get('end', 0)
                })
            
            return dict(grouped)
        
        except Exception as e:
            return {'error': f'Entity extraction failed: {str(e)}'}
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract legal-specific entities using pattern matching + NER
        
        Args:
            text: Legal document text
        
        Returns:
            Dictionary with legal entities
        """
        entities = {
            'judges': [],
            'courts': [],
            'parties': [],
            'dates': [],
            'statutes': [],
            'case_numbers': [],
            'citations': []
        }
        
        # 1. Extract Judges
        judge_patterns = [
            r"Justice\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"Hon'ble\s+Justice\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"Chief Justice\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        ]
        
        for pattern in judge_patterns:
            matches = re.findall(pattern, text)
            entities['judges'].extend([f"Justice {m}" for m in matches])
        
        # 2. Extract Courts
        court_patterns = [
            r"(Supreme Court of India)",
            r"([\w\s]+High Court)",
            r"(District Court of [\w\s]+)",
            r"([\w\s]+District Court)"
        ]
        
        for pattern in court_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['courts'].extend(matches)
        
        # 3. Extract Dates
        date_patterns = [
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b"
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['dates'].extend(matches)
        
        # 4. Extract Statutes
        statute_patterns = [
            r"(Section\s+\d+[A-Z]?\s+of\s+[\w\s]+Act)",
            r"(Article\s+\d+[A-Z]?)",
            r"([\w\s]+Act,?\s+\d{4})"
        ]
        
        for pattern in statute_patterns:
            matches = re.findall(pattern, text)
            entities['statutes'].extend(matches)
        
        # 5. Extract Case Numbers
        case_number_patterns = [
            r"(Civil Appeal No\.\s*\d+\s+of\s+\d{4})",
            r"(Writ Petition No\.\s*\d+\s+of\s+\d{4})",
            r"(SLP\(C\)\s*No\.\s*\d+\s+of\s+\d{4})"
        ]
        
        for pattern in case_number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['case_numbers'].extend(matches)
        
        # 6. Extract Citations
        citation_patterns = [
            r"(AIR\s+\d{4}\s+\w+\s+\d+)",
            r"(\d{4}\s+SCC\s+\d+)",
            r"(\(\d{4}\)\s+\d+\s+SCC\s+\d+)"
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            entities['citations'].extend(matches)
        
        # 7. Use NER for parties (persons and organizations)
        ner_entities = self.extract_entities(text)
        
        if 'PER' in ner_entities:
            entities['parties'].extend([e['text'] for e in ner_entities['PER']])
        
        if 'ORG' in ner_entities:
            # Filter out courts from organizations
            orgs = [e['text'] for e in ner_entities['ORG']]
            parties = [o for o in orgs if 'court' not in o.lower()]
            entities['parties'].extend(parties)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def get_entity_summary(self, text: str) -> str:
        """
        Get a formatted summary of extracted entities
        
        Args:
            text: Legal document text
        
        Returns:
            Formatted string with entity summary
        """
        entities = self.extract_legal_entities(text)
        
        summary_lines = []
        
        if entities['judges']:
            summary_lines.append(f"👨‍⚖️ Judges: {', '.join(entities['judges'][:3])}")
        
        if entities['courts']:
            summary_lines.append(f"🏛️ Courts: {', '.join(entities['courts'][:2])}")
        
        if entities['parties']:
            summary_lines.append(f"👥 Parties: {', '.join(entities['parties'][:3])}")
        
        if entities['dates']:
            summary_lines.append(f"📅 Dates: {', '.join(entities['dates'][:3])}")
        
        if entities['statutes']:
            summary_lines.append(f"📜 Statutes: {', '.join(entities['statutes'][:2])}")
        
        if entities['citations']:
            summary_lines.append(f"📚 Citations: {', '.join(entities['citations'][:2])}")
        
        if not summary_lines:
            return "No entities extracted"
        
        return '\n'.join(summary_lines)

# Test function
def test_ner():
    """Test the legal NER system"""
    
    print("=" * 80)
    print("LEGAL NER TEST")
    print("=" * 80)
    
    ner = LegalNER()
    
    # Sample legal text
    sample_text = """
    In the case of Kesavananda Bharati v. State of Kerala, decided on 24th April 1973,
    the Supreme Court of India established the basic structure doctrine. The bench was
    presided over by Chief Justice S.M. Sikri along with Justice Shelat and Justice Grover.
    The case involved interpretation of Article 368 of the Constitution of India.
    The petitioner, Kesavananda Bharati, challenged the Kerala Land Reforms Act, 1963.
    The judgment cited AIR 1967 SC 1643 and (1973) 4 SCC 225. This landmark decision
    was delivered in Civil Appeal No. 135 of 1970.
    """
    
    print("\n📄 Sample Text:")
    print(sample_text[:200] + "...")
    
    # Test 1: Extract all entities
    print("\n🔍 Test 1: Extract Legal Entities")
    entities = ner.extract_legal_entities(sample_text)
    
    for entity_type, values in entities.items():
        if values:
            print(f"\n   {entity_type.upper()}:")
            for value in values[:3]:  # Show first 3
                print(f"      • {value}")
    
    # Test 2: Entity summary
    print("\n🔍 Test 2: Entity Summary")
    summary = ner.get_entity_summary(sample_text)
    print(summary)
    
    # Test 3: Standard NER
    print("\n🔍 Test 3: Standard NER Entities")
    standard_entities = ner.extract_entities(sample_text)
    for entity_type, values in standard_entities.items():
        if entity_type != 'error':
            print(f"\n   {entity_type}:")
            for entity in values[:3]:
                print(f"      • {entity['text']} (confidence: {entity['confidence']})")
    
    print("\n" + "=" * 80)
    print("✅ NER tests complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_ner()
