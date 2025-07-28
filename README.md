# 📄 PDF Document Outline Extractor

## 📚 Overview

This project implements a solution for **Round 1A: Understand Your Document Challenge** - a PDF outline extraction system that automatically identifies and extracts document structure including titles and hierarchical headings (H1, H2, H3) with their corresponding page numbers.

## 🎯 Challenge Description

The goal is to build a machine-readable document structure extractor that:

- 📄 Processes PDF files up to 50 pages  
- 🏷 Extracts document title and headings (H1, H2, H3)  
- 💾 Outputs structured JSON with heading levels and page numbers  
- ⚡ Runs efficiently within strict performance constraints  

## 🚀 Our Approach

### 🧩 Methodology

Our solution employs a multi-layered approach to PDF structure extraction:

1. **Text Extraction**: Utilize robust PDF parsing libraries to extract text while preserving formatting information  
2. **Layout Analysis**: Analyze font sizes, styles, positioning, and whitespace patterns to identify heading candidates  
3. **Hierarchy Detection**: Implement Random Forest classifier to determine heading levels based on multiple factors beyond just font size  
4. **Context Validation**: Apply contextual rules to validate and refine heading classifications  

### ✨ Key Features

- 🔤 **Font-agnostic Detection**: Does not rely solely on font sizes for heading identification  
- 🔍 **Multi-factor Analysis**: Considers position, styling, spacing, and content patterns  
- 🛠 **Robust Parsing**: Handles various PDF formats and layouts  
- ⏱ **Performance Optimized**: Designed for sub-10-second execution on 50-page documents  
- 🌐 **Multilingual Support**: Capable of processing documents in multiple languages and various character encodings, ensuring broad applicability across diverse document sets. 

## 🧰 Technology Stack

### Core Libraries

- **PDF Processing**: [Specify your PDF library - e.g., PyMuPDF, pdfplumber, etc.]  
- **Text Analysis**: [Specify libraries used for text processing]  
- **Layout Detection**: [Specify any computer vision or layout analysis tools]  

### 🤖 Machine Learning Model

- **Final Model**: **Random Forest Classifier**  
- 📦 **Model Size**: < 200MB (compliant with constraints)  
- 🏛 **Architecture**: Ensemble-based classification for heading level detection  
- 🔌 **Offline Operation**: All models bundled within container  

## 🛠 Development Challenges & Problems Faced

During the development process, we encountered several significant challenges that required iterative solutions and model improvements:

### 1. Decision Tree Classification Issues

**Problem**: Our initial implementation using decision tree models showed promising overall accuracy, but suffered from systematic confusion between H1 and H2 heading levels.

**Impact**:

- ✔️ Correct identification of headings but incorrect hierarchy assignment  
- 🔄 Resulted in flattened or inverted document structure  
- ❌ Affected downstream processing and JSON output quality  

**Solution**: Enhanced feature engineering and eventually migrated to Random Forest to leverage ensemble learning for better hierarchy discrimination.

### 2. Model Evolution: Decision Tree → Random Forest

**Problem**: Single decision tree models were prone to overfitting and couldn't capture the complex relationships between different heading features.

**Impact**:  

- ❗ Inconsistent performance across different PDF types  
- 🚫 Poor generalization to unseen document formats  

**Solution**: **Implemented Random Forest classifier as our final model**, which provided:

- 🔗 Better handling of feature interactions  
- 🛡 Improved robustness against overfitting  
- 🎯 Enhanced accuracy in distinguishing between H1, H2, and H3 levels  
- 📈 More stable performance across diverse document types  

### 3. Title Extraction from Metadata

**Problem**: The title extraction mechanism was pulling information from PDF metadata rather than analyzing the actual document content.

**Impact**:  

- 📛 Extracted titles were often generic, unrelated, or missing entirely  
- ❌ No correlation between extracted title and actual document content  
- 😕 Failed to identify dynamically generated or embedded titles  

**Solution**: Implemented **content-based title detection** by analyzing the first page layout, font hierarchies, and positioning patterns.

### 4. False Positive Heading Detection

**Problem**: Bold body text and emphasized content were being incorrectly classified as headings.

**Impact**:

- ➕ Inflated heading counts in output JSON  
- 🔄 Incorrect document structure representation  
- 📉 Reduced precision in heading detection  

**Solution**: Added contextual validation rules and trained the Random Forest model with negative examples including:

- 🚦 Minimum/maximum heading length thresholds  
- 📍 Positional constraints (e.g., not mid-paragraph)  
- 🔍 Content pattern analysis (avoiding inline emphasis)

### 5. DistilBERT Integration Issues

**Problem**: When attempting to use DistilBERT for semantic heading classification, the punkt tokenizer library caused significant compatibility and performance issues.

**Impact**:

- 🚫 Tokenization failures on certain PDF text extractions  
- 🐢 Increased processing time beyond acceptable limits  
- 🔄 Dependency conflicts in containerized environment  

**Solution**:

- ❌ Abandoned the DistilBERT approach due to complexity and performance constraints  
- 🔄 Switched to Random Forest with carefully engineered features  
- 🛡 Implemented fallback mechanisms for problematic text segments  
- ⚙️ Optimized model loading and inference pipeline  

### 6. Performance vs. Accuracy Trade-offs

**Problem**: Balancing the sophisticated analysis required for accurate heading detection with the strict 10-second execution time constraint.

**Impact**:

- 🕰 Initial implementations exceeded time limits  
- ⚖️ Had to compromise on some advanced features  

**Solution**: Random Forest provided an optimal balance of accuracy and speed, enabling comprehensive feature analysis within time constraints.

## Project Structure


```
├── Dockerfile
├── README.md
├── requirements.txt
├── process_pdfs.py
└──  Model_randomforest.pkl

```

0
## Installation \& Usage

### Building the Docker Image

```bash
docker build --platform linux/amd64 -t pdf-outline-extractor:latest .
```


### Running the Solution

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  pdf-outline-extractor:latest
```


### Input/Output Format

**Input**: PDF files in `/app/input` directory

**Output**: JSON files in `/app/output` directory with format:

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Chapter 1: Introduction",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "Background",
      "page": 2
    },
    {
      "level": "H3",
      "text": "Historical Context",
      "page": 3
    }
  ]
}
```



## 📊 Performance Specifications

| Metric          | Specification                 | Our Performance         |
|-----------------|------------------------------|------------------------|
| ⏱ Execution Time | ≤ 10 seconds (50-page PDF)    | [Your actual performance] |
| 📦 Model Size    | ≤ 200MB                      | [Your model size]      |
| ⚙️ Architecture | AMD64 CPU only                | ✅ Compliant            |
| 💾 Memory       | 16GB RAM available            | [Your memory usage]    |
| 🌐 Network      | Offline operation             | ✅ No network calls    |

## 🧠 Algorithm Details

### Random Forest Heading Classification

1. 📝 **Text Extraction with Formatting**: Extract text while preserving font information, positioning, and styling  
2. ⚙️ **Feature Engineering**: Calculate comprehensive features for Random Forest input:  
    - 🔠 Font size relative to document average  
    - 🅱️ Bold/italic styling indicators  
    - 📍 Position on page (top, middle, bottom)  
    - ↔️ Whitespace before/after ratios  
    - 🔡 Line length and capitalization patterns  
    - 📊 Text density and character count  
    - ↪️ Indentation levels  
3. 🌳 **Random Forest Classification**: Multi-class classification (Title/H1/H2/H3/Body) using ensemble of decision trees  
4. 🔍 **Hierarchy Validation**: Post-processing to ensure logical heading hierarchy and resolve conflicts  

### Model Training Features

The Random Forest model was trained on the following key features:

- ✍️ **Typography Features**: Font size, weight, style  
- 📏 **Layout Features**: Position, spacing, alignment  
- 🧾 **Content Features**: Text length, capitalization, punctuation  
- 🏞 **Context Features**: Surrounding text properties, page location  

### Special Considerations

- 🌍 **Multi-language Support**: Handles various character encodings and languages  
- 📐 **Complex Layouts**: Robust against multi-column layouts, tables, and graphics  
- 🚧 **Edge Cases**: Handles PDFs with inconsistent formatting or missing structure  

## ✅ Testing & Validation

The Random Forest solution has been tested on:

- 🧑🎓 Simple academic papers  
- 🛠 Complex technical documents  
- 🌐 Multi-language documents  
- 🖨 Various PDF generators and formats  

## ⚠️ Known Limitations

- 📉 Model performance depends on training data diversity  
- 🔄 May require retraining for highly specialized document types  
- ⚠️ Feature extraction sensitive to PDF generation quality  

## 🔮 Future Enhancements (Round 1B Preparation)

This modular design enables easy extension for:

- 🧠 Enhanced semantic understanding  
- 📝 Content summarization  
- 🔗 Cross-document relationship detection  
- 🌏 Advanced multilingual processing  
- 🎯 Model fine-tuning for specific document domains  

## 💡 Development Notes

### Model Training

- 🌳 Random Forest trained on diverse PDF corpus  
- 🔍 Cross-validation used for hyperparameter tuning  
- 📊 Feature importance analysis guided feature selection  

### Debugging

- 🔄 Add `--volume` mount for logs if needed during development  
- ⚙️ Use environment variables for configuration in development  
- 📉 Model prediction confidence scores available for debugging  

### Optimization Techniques

- ⚡ Efficient feature extraction pipeline  
- 📦 Model compression for size constraints  
- 🤹‍♂️ Parallel processing where applicable  

## 🛡 Compliance Checklist

- ✅ AMD64 architecture compatibility  
- ✅ No GPU dependencies  
- ✅ Model size ≤ 200MB  
- ✅ Offline operation (no network calls)  
- ✅ Execution time ≤ 10 seconds for 50-page PDF  
- ✅ Proper Docker volume mounting  
- ✅ JSON output format compliance  

---
