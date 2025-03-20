# Intelligent Knowledge Base

An advanced RAG (Retrieval-Augmented Generation) system that automatically selects optimal parsing methods and embedding models based on document types and content.

## Features

- **Intelligent Content Analysis**: Automatically detects document type and content characteristics
- **Adaptive Processing**: Selects optimal document splitters and embedding models based on content type
- **Format-Specific Optimization**: Special handling for PDFs, code files, technical documents, and more
- **Robust Conversion**: Handles a wide variety of document formats (PDF, DOCX, HTML, TXT, etc.)
- **Interactive Interface**: User-friendly Chainlit interface for uploading documents and asking questions

## Document Type Specialization

The system automatically optimizes processing based on content type:

- **Code**: Uses code-optimized embedding models and word-based splitting
- **PDF**: Uses PDF-specific processing with passage-based splitting
- **Technical**: Uses models optimized for technical content with denser splitting
- **Text**: General purpose processing for standard text documents

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your API key:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the knowledge base application:

```
chainlit run knowlage.py
```

Then:
1. Upload any document (PDF, TXT, HTML, DOCX, etc.)
2. Wait for processing (the system will analyze and index the document)
3. Ask questions about the document content

## How It Works

1. **Content Analysis**: Analyzes documents using MIME types and extensions
2. **Converter Selection**: Chooses optimal converter based on file type
3. **Embedding Model Selection**: Selects specialized embedding models for different content types
4. **Document Splitting**: Configures splitters optimally for different content types
5. **RAG Pipeline**: Builds a specialized RAG pipeline tailored to the content
6. **Query Processing**: Processes user queries with content-aware retrieval and generation

## Supported File Types

- Text files (.txt)
- PDF documents (.pdf)
- Microsoft Word (.docx)
- HTML (.html, .htm)
- Markdown (.md)
- JSON (.json)
- CSV (.csv)
- PowerPoint (.pptx)
- Excel (.xlsx)
- Programming code files (.py, .js, .java, etc.)
- And many more formats...

## Advanced Configuration

To customize the embedding models or splitter configurations, modify the `EMBEDDING_MODELS` and `SPLITTER_CONFIG` dictionaries in `knowlage.py`.
