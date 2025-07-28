import os
import json
import pandas as pd
import fitz  # pymupdf
import nltk
import joblib
from collections import Counter
import re
import math
from pathlib import Path

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

class FinalHeadingDetector:
    def __init__(self, model_path='heading_detection_model.pkl'):
        """Initialize the heading detector with trained model"""
        print(model_path)
        print("🤖 Loading trained model...")
        try:
            self.model = joblib.load(model_path)
            print("✅ Model loaded successfully!")
        except FileNotFoundError:
            print(f"❌ Error: Model file '{model_path}' not found!")
            raise
        
        # Feature names in exact order used during training
        self.feature_names = [
            'bold_or_not', 'font_threshold_flag', 'words', 'text_case',
            'verbs', 'nouns', 'cardinal_numbers', 'font_size', 
            'position_score', 'page_number', 'relative_font_size'
        ]
        print("debug")
        # Class mapping
        self.class_to_level = {
            0: 'Body',
            1: 'H3',
            2: 'H2', 
            3: 'H1',
            4: 'Title'
        }
        
        # Enumeration pattern regex
        self.enum_pattern = re.compile(r'^(\d+|[IVXLCDM]+|[a-zA-Z])[\.\)\:\-]\s*')

    def extract_pdf_metadata_title(self, pdf_path):
        """Extract title from PDF metadata"""##may remove345
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            
            title = metadata.get('title', '').strip()
            
            if title:
                # Clean common metadata noise
                title = re.sub(r'^(Microsoft Word - |untitled)', '', title, flags=re.IGNORECASE)
                title = re.sub(r'\.(doc|docx|pdf|txt)$', '', title, flags=re.IGNORECASE)
                title = title.strip()
            
            return title if title else None
            
        except:
            return None

    def normalize_text(self, text):
        """Normalize text for better feature extraction"""
        text = text.strip()
        
        # Handle enumeration patterns like "1.Hello" → "1. Hello"
        if self.enum_pattern.match(text):
            # Add space after enumeration marker if missing
            text = self.enum_pattern.sub(lambda m: m.group(1) + '. ', text)
            # Clean up multiple spaces
            text = re.sub(r'\s+', ' ', text)
        
        # Handle common formatting issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces → single space
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Fix sentence spacing
        
        return text.strip()

    def detect_drop_caps(self, spans):
        """Detect drop caps with enhanced criteria"""
        if not spans:
            return set()
        
        drop_cap_indices = set()
        
        # Calculate font statistics
        font_sizes = [span['font_size'] for span in spans]
        if not font_sizes:
            return drop_cap_indices
            
        font_counter = Counter(font_sizes)
        most_common_font = font_counter.most_common(1)[0][0]  # Body text font
        avg_font_size = sum(font_sizes) / len(font_sizes)
        
        print(f"   📊 Font analysis: avg={avg_font_size:.1f}, common={most_common_font}")
        
        for i, span in enumerate(spans):
            text = span['text'].strip()
            font_size = span['font_size']
            
            # Enhanced drop cap detection criteria
            drop_cap_score = 0
            reasons = []
            
            # 1. Very short text (1-3 characters)
            if 1 <= len(text) <= 3:
                drop_cap_score += 2
                reasons.append("short_text")
                
                # 2. Font size significantly larger than body text
                if font_size >= most_common_font * 2:
                    drop_cap_score += 3
                    reasons.append("large_font")
                    
                    # 3. Single letter or decorative character
                    if len(text) == 1 and text.isalpha():
                        drop_cap_score += 2
                        reasons.append("single_letter")
                    
                    # 4. Position check - not at very bottom
                    if span['position_score'] > 0.3:
                        drop_cap_score += 1
                        reasons.append("good_position")
                        
                        # 5. Check if followed by normal text
                        if i + 1 < len(spans):
                            next_span = spans[i + 1]
                            if (len(next_span['text'].strip()) > 5 and 
                                next_span['font_size'] < font_size * 0.7 and
                                span['page_num'] == next_span['page_num']):
                                drop_cap_score += 2
                                reasons.append("followed_by_body")
            
            # 6. Special patterns that are likely drop caps
            if (text.isupper() and len(text) == 1 and 
                font_size >= most_common_font * 2.5):
                drop_cap_score += 3
                reasons.append("decorative_cap")
            
            # Decision: need score >= 6 for drop cap classification
            if drop_cap_score >= 6:
                drop_cap_indices.add(i)
                print(f"   🎯 Drop cap detected: '{text}' (font={font_size}, score={drop_cap_score}, reasons={reasons})")
        
        print(f"   ✅ Found {len(drop_cap_indices)} drop caps")
        return drop_cap_indices

    def merge_drop_caps_with_following_text(self, spans, drop_cap_indices):
        """Merge drop caps with their following paragraph text"""
        if not drop_cap_indices:
            return spans
        
        merged_spans = []
        skip_indices = set()
        
        for i, span in enumerate(spans):
            if i in skip_indices:
                continue
                
            if i in drop_cap_indices:
                # This is a drop cap, try to merge with following text
                drop_cap_text = span['text'].strip()
                merged_text = drop_cap_text
                merge_span = span.copy()
                
                # Look for following text to merge (check next 3 spans)
                for j in range(i + 1, min(i + 4, len(spans))):
                    following_span = spans[j]
                    
                    # Should be on same page
                    if following_span['page_num'] == span['page_num']:
                        # Should have smaller font (body text)
                        if following_span['font_size'] < span['font_size'] * 0.8:
                            # Should have reasonable length
                            following_text = following_span['text'].strip()
                            if len(following_text) > 10:
                                # Merge the texts
                                merged_text = drop_cap_text + following_text
                                # Use following span's properties (body text properties)
                                merge_span = following_span.copy()
                                merge_span['text'] = merged_text
                                skip_indices.add(j)
                                print(f"   🔗 Merged drop cap '{drop_cap_text}' with following text")
                                break
                
                merged_spans.append(merge_span)
            else:
                merged_spans.append(span)
        
        return merged_spans

    def get_bold_status_fitz(self, span):
        """Extract bold status using fitz library with multiple methods"""
        font_name = span.get("font", "").lower()
        font_flags = span.get("flags", 0)
        
        # Method 1: Font name analysis
        bold_indicators = ['bold', 'black', 'heavy', 'semibold', 'demibold', 'extrabold']
        name_bold = any(indicator in font_name for indicator in bold_indicators)
        
        # Method 2: Font flags (bit 4 indicates bold)
        flag_bold = bool(font_flags & 2**4)
        
        # Method 3: Font name suffixes
        bold_suffixes = ['-bold', '-b', '-black', '-heavy', '-sb', '-db']
        suffix_bold = any(font_name.endswith(suffix) for suffix in bold_suffixes)
        
        # Method 4: Font weight in name
        weight_bold = any(weight in font_name for weight in ['700', '800', '900'])
        
        return 1 if (name_bold or flag_bold or suffix_bold or weight_bold) else 0

    def calculate_document_thresholds(self, font_sizes):
        """Calculate font size thresholds for the document"""
        if not font_sizes:
            return {'body_threshold': 12}
        
        # Use the most common font size as body text baseline
        font_counter = Counter(font_sizes)
        most_common_size = font_counter.most_common(1)[0][0]
        
        return {'body_threshold': most_common_size}

    def get_text_case(self, text):
        """Determine text case classification"""
        text = text.strip()
        if not text:
            return 3
        
        # Remove enumeration prefix for case analysis
        text_for_case = self.enum_pattern.sub('', text).strip()
        if not text_for_case:
            return 3
        
        if text_for_case.islower():
            return 0  # lower case
        elif text_for_case.isupper():
            return 1  # upper case
        elif text_for_case.istitle():
            return 2  # title case
        else:
            return 3  # sentence case or mixed

    def pos_tag_features(self, text):
        """Extract POS tagging features with improved handling"""
        try:
            # Remove enumeration prefix for POS analysis
            text_for_pos = self.enum_pattern.sub('', text).strip()
            if not text_for_pos:
                return 0, 0, 0
            
            tokens = nltk.word_tokenize(text_for_pos.lower())
            pos_tags = nltk.pos_tag(tokens)
            
            verbs = sum(1 for word, tag in pos_tags if tag.startswith('VB'))
            nouns = sum(1 for word, tag in pos_tags if tag.startswith('NN'))
            cardinal_numbers = sum(1 for word, tag in pos_tags if tag == 'CD')
            
            return verbs, nouns, cardinal_numbers
        except:
            return 0, 0, 0

    def extract_text_features(self, pdf_path):
        """Extract all text segments with comprehensive preprocessing"""
        doc = fitz.open(pdf_path)
        all_spans = []
        
        print(f"   📖 Extracting text from PDF...")
        
        # First pass: collect all text spans
        for page_num, page in enumerate(doc):
            page_height = page.rect.height
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            raw_text = span["text"].strip()
                            if raw_text and len(raw_text) >= 1:
                                # Normalize text
                                normalized_text = self.normalize_text(raw_text)
                                
                                if normalized_text:
                                    position_score = 1 - (span["bbox"][1] / page_height)
                                    
                                    all_spans.append({
                                        'text': normalized_text,
                                        'font_size': int(span["size"]),
                                        'span_data': span,
                                        'page_num': page_num,
                                        'position_score': position_score
                                    })
        
        if not all_spans:
            doc.close()
            return []
        
        print(f"   📊 Extracted {len(all_spans)} text segments")
        
        # Detect and handle drop caps
        drop_cap_indices = self.detect_drop_caps(all_spans)
        if drop_cap_indices:
            all_spans = self.merge_drop_caps_with_following_text(all_spans, drop_cap_indices)
            print(f"   🔗 After drop cap handling: {len(all_spans)} segments")
        
        # Filter out remaining noise
        filtered_spans = []
        for span in all_spans:
            text = span['text'].strip()
            # Keep meaningful segments
            if (len(text) >= 3 or 
                (len(text) >= 1 and self.enum_pattern.match(text)) or
                (len(text) >= 2 and span['font_size'] <= 24)):
                filtered_spans.append(span)
        
        all_spans = filtered_spans
        print(f"   ✅ Final segments after filtering: {len(all_spans)}")
        
        # Calculate document thresholds and relative sizes
        font_sizes = [span['font_size'] for span in all_spans]
        thresholds = self.calculate_document_thresholds(font_sizes)
        
        max_font = max(font_sizes) if font_sizes else 12
        min_font = min(font_sizes) if font_sizes else 12
        font_range = max_font - min_font if max_font > min_font else 1
        
        # Extract features for each span
        text_features = []
        
        for span in all_spans:
            text = span['text']
            font_size = span['font_size']
            span_data = span['span_data']
            page_num = span['page_num']
            position_score = span['position_score']
            
            # Extract all 11 features
            bold_or_not = self.get_bold_status_fitz(span_data)
            font_threshold_flag = 1 if font_size > thresholds['body_threshold'] else 0
            words = len(text.split())
            text_case = self.get_text_case(text)
            verbs, nouns, cardinal_numbers = self.pos_tag_features(text)
            relative_font_size = (font_size - min_font) / font_range
            
            # Create feature vector in exact training order
            features = [
                bold_or_not, font_threshold_flag, words, text_case,
                verbs, nouns, cardinal_numbers, font_size, 
                position_score, page_num, relative_font_size
            ]
            
            text_features.append({
                'text': text,
                'page_num': page_num + 1,  # Convert to 1-based page numbering
                'features': features,
                'position_score': position_score,
                'font_size': font_size
            })
        
        doc.close()
        return text_features

    def predict_headings(self, text_features):
        """Predict heading levels with intelligent post-processing"""
        if not text_features:
            return []
        
        # Prepare feature matrix
        feature_matrix = [item['features'] for item in text_features]
        
        # Make predictions
        predictions = self.model.predict(feature_matrix)
        probabilities = self.model.predict_proba(feature_matrix)
        
        # Process results with intelligent filtering
        results = []
        for i, item in enumerate(text_features):
            prediction = predictions[i]
            confidence = probabilities[i].max()
            text = item['text'].strip()
            
            # Intelligent post-processing
            should_include = True
            adjusted_prediction = prediction
            
            # Handle enumeration patterns intelligently
            if self.enum_pattern.match(text):
                # If model says Body but has enumeration, check if it should be upgraded
                if prediction == 0 and confidence < 0.7:
                    # Look at second-best prediction
                    sorted_probs = probabilities[i].argsort()
                    second_best = sorted_probs[-2]
                    if second_best in [1, 2, 3] and probabilities[i][second_best] > 0.3:
                        adjusted_prediction = second_best
                        print(f"   🔄 Upgraded enumeration '{text[:30]}...' from Body to {self.class_to_level[second_best]}")
            
            # Filter out obvious noise
            if len(text) <= 2 and adjusted_prediction > 0:
                if confidence < 0.8 and not self.enum_pattern.match(text):
                    should_include = False
            
            # Filter single characters unless very high confidence
            if len(text) == 1 and confidence < 0.9:
                should_include = False
            
            # Only include headings and titles (filter out body text)
            if adjusted_prediction > 0 and should_include:
                results.append({
                    'text': text,
                    'page': item['page_num'],
                    'level': self.class_to_level[adjusted_prediction],
                    'confidence': round(confidence, 3),
                    'position_score': item['position_score'],
                    'font_size': item['font_size']
                })
        
        return results

    def find_document_title(self, headings, metadata_title):
        """Find document title using prioritized strategies with deduplication"""
        title_candidates = []
        candidate_scores = {}  # Track confidence scores for each candidate
        
        print("   📚 Finding document title...")
        
        # Strategy 1: Predicted titles (highest priority)
        predicted_titles = [h for h in headings if h['level'] == 'Title']
        if predicted_titles:
            predicted_titles.sort(key=lambda x: (-x['position_score'], x['page']))
            for title in predicted_titles:
                text = title['text'].strip()
                if text not in candidate_scores:
                    title_candidates.append(text)
                    candidate_scores[text] = 100 + title['position_score'] * 10 + title.get('confidence', 0) * 10
        
        # Strategy 2: Fallback to first H1 on first page
        first_page_h1s = [h for h in headings if h['level'] == 'H1' and h['page'] == 1]
        if first_page_h1s:
            first_page_h1s.sort(key=lambda x: -x['position_score'])
            text = first_page_h1s[0]['text'].strip()
            if text not in candidate_scores:
                title_candidates.append(text)
                candidate_scores[text] = 80 + first_page_h1s[0]['position_score'] * 5
        
        # Strategy 3: Smart partial matching with metadata
        if metadata_title:
            metadata_words = set(word.lower() for word in metadata_title.split() if len(word) > 3)
            
            for heading in headings:
                if heading['level'] in ['H1', 'H2'] and len(heading['text']) > 10:
                    heading_words = set(word.lower() for word in heading['text'].split() if len(word) > 3)
                    
                    # Require significant overlap (at least 50% of metadata words)
                    overlap = len(metadata_words.intersection(heading_words))
                    if overlap >= len(metadata_words) * 0.5 and overlap >= 2:
                        text = heading['text'].strip()
                        if text not in candidate_scores:
                            title_candidates.append(text)
                            candidate_scores[text] = 70 + (overlap / len(metadata_words)) * 15
                        break  # Take first good match only
        
        # Strategy 4: Metadata title (fallback priority)
        if metadata_title and metadata_title.strip():
            clean_metadata = metadata_title.strip()
            if clean_metadata not in candidate_scores:
                title_candidates.append(clean_metadata)
                candidate_scores[clean_metadata] = 60  # Lower priority as fallback
        
        print(f"   🎯 Title candidates: {len(title_candidates)}")
        for candidate in title_candidates:
            print(f"      '{candidate[:40]}...' (score: {candidate_scores[candidate]:.1f})")
        
        # Select best candidate based on weighted scoring
        if title_candidates:
            def title_quality_score(candidate):
                base_score = candidate_scores[candidate]
                length_bonus = min(len(candidate) / 10, 10)
                word_count_bonus = min(len(candidate.split()), 5)
                
                # Penalty for very short or very long titles
                if len(candidate) < 10:
                    length_penalty = (10 - len(candidate)) * 2
                elif len(candidate) > 100:
                    length_penalty = (len(candidate) - 100) * 0.5
                else:
                    length_penalty = 0
                
                return base_score + length_bonus + word_count_bonus - length_penalty
            
            best_title = max(title_candidates, key=title_quality_score)
            final_score = title_quality_score(best_title)
            print(f"   ✅ Selected title: '{best_title}' (final score: {final_score:.1f})")
            return best_title
        
        return "Untitled Document"


    def create_outline(self, headings):
        """Create clean document outline structure"""
        outline = []
        
        # Filter out titles from outline (they go in title field)
        heading_levels = [h for h in headings if h['level'] != 'Title']
        
        # Sort by page number, then by position on page
        heading_levels.sort(key=lambda x: (x['page'], -x['position_score']))
        
        # Build outline with quality checks
        for heading in heading_levels:
            text = heading['text'].strip()
            
            # Skip very short headings unless they look like section numbers
            if len(text) < 3 and not self.enum_pattern.match(text):
                continue
                
            outline.append({
                'level': heading['level'],
                'text': text,
                'page': heading['page']-1 #0 based indexing in output
            })
        
        return outline

    def process_pdf(self, pdf_path):
        """Process a single PDF and return structured JSON"""
        print(f"📄 Processing: {os.path.basename(pdf_path)}")
        
        try:
            # Extract metadata title
            metadata_title = self.extract_pdf_metadata_title(pdf_path)
            if metadata_title:
                print(f"   📋 Metadata title: '{metadata_title}'")
            
            # Extract text features with all preprocessing
            text_features = self.extract_text_features(pdf_path)
            
            if not text_features:
                print(f"   ⚠️  No text found in PDF")
                return {
                    "title": metadata_title or "Untitled 69 Document",
                    "outline": []
                }
            
            # Predict headings
            headings = self.predict_headings(text_features)
            
            if not headings:
                print(f"   ⚠️  No headings detected")
                return {
                    "title": metadata_title or "Untitled  ula Document", 
                    "outline": []
                }
            
            print(f"   ✅ Found {len(headings)} headings")
            
            # Find document title
            document_title = self.find_document_title(headings, metadata_title)
            
            # Create outline
            outline = self.create_outline(headings)
            
            # Show processing summary
            level_counts = Counter([h['level'] for h in headings])
            outline_counts = Counter([item['level'] for item in outline])
            print(f"   📊 Detected: {dict(level_counts)}")
            print(f"   📋 Outline: {dict(outline_counts)}")
            print(f"   📖 Title: '{document_title[:50]}{'...' if len(document_title) > 50 else ''}'")
            
            return {
                "title": document_title,
                "outline": outline
            }
            
        except Exception as e:
            print(f"   ❌ Error processing PDF: {e}")
            import traceback
            traceback.print_exc()
            return None

def process_folder(input_folder, output_folder, model_path='heading_detection_model.pkl'):
    """Process all PDFs in input folder and save JSON files to output folder"""
    
    # Initialize detector
    try:
        detector = FinalHeadingDetector(model_path)
    except:
        return
    
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created output folder: {output_folder}")
    
    # Check input folder
    if not os.path.exists(input_folder):
        print(f"❌ Input folder '{input_folder}' not found!")
        return
    
    # Get PDF files
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in '{input_folder}'")
        return
    
    print(f"🚀 Processing {len(pdf_files)} PDF files with COMPLETE enhancement...")
    print("=" * 70)
    
    successful = 0
    failed = 0
    total_headings = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        
        # Process PDF
        result = detector.process_pdf(pdf_path)
        
        if result:
            # Save JSON file
            json_filename = os.path.splitext(pdf_file)[0] + '.json'
            json_path = os.path.join(output_folder, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            outline_count = len(result.get('outline', []))
            total_headings += outline_count
            
            print(f"   💾 Saved: {json_filename} ({outline_count} headings)")
            successful += 1
        else:
            print(f"   ❌ Failed to process: {pdf_file}")
            failed += 1
        
        print()
    
    print("=" * 70)
    print(f"🎉 Processing Complete!")
    print(f"📊 Results: {successful} successful, {failed} failed")
    print(f"📈 Total headings extracted: {total_headings}")
    print(f"📁 JSON files saved in: {output_folder}")
    
    if successful > 0:
        print(f"📊 Average headings per document: {total_headings/successful:.1f}")

def main():
    """Main function with comprehensive PDF processing"""
    print("🎯 FINAL PDF Heading Detection System")
    print("✨ Features: Drop caps, Enumeration, Smart classification")
    print("=" * 60)
    
    # Configuration
    input_folder = Path("/app/input")
    output_folder = Path("/app/output")
    if not input_folder.exists():
        raise FileNotFoundError("Input directory /input does not exist")
    output_folder.mkdir(parents=True, exist_ok=True)
    model_path = 'model_RandomForest.pkl'
    
    # Check model
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        print("Please train the model first using the training script.")
        return
    
    # Create input folder if needed
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"📁 Created input folder: {input_folder}")
        print("Please add your PDF files and run again.")
        return
    
    # Process all PDFs
    process_folder(input_folder, output_folder, model_path)

if __name__ == "__main__":
    main()
