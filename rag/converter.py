'''
| Converter                                                                                                                                                      | Description                                                                                                   |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| [AzureOCRDocumentConverter](https://docs.haystack.deepset.ai/docs/azureocrdocumentconverter "https://docs.haystack.deepset.ai/docs/azureocrdocumentconverter") | Converts PDF (both searchable and image-only), JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, and HTML to documents. |
| [CSVToDocument](https://docs.haystack.deepset.ai/docs/csvtodocument "https://docs.haystack.deepset.ai/docs/csvtodocument")                                     | Converts CSV files to documents.                                                                              |
| [DOCXToDocument](https://docs.haystack.deepset.ai/docs/docxtodocument "https://docs.haystack.deepset.ai/docs/docxtodocument")                                  | Convert DOCX files to documents.                                                                              |
| [HTMLToDocument](https://docs.haystack.deepset.ai/docs/htmltodocument "https://docs.haystack.deepset.ai/docs/htmltodocument")                                  | Converts HTML files to documents.                                                                             |
| [JSONConverter](https://docs.haystack.deepset.ai/docs/jsonconverter "https://docs.haystack.deepset.ai/docs/jsonconverter")                                     | Converts JSON files to text documents.                                                                        |
| [MarkdownToDocument](https://docs.haystack.deepset.ai/docs/markdowntodocument "https://docs.haystack.deepset.ai/docs/markdowntodocument")                      | Converts markdown files to documents.                                                                         |
| [OpenAPIServiceToFunctions](https://docs.haystack.deepset.ai/docs/openapiservicetofunctions "https://docs.haystack.deepset.ai/docs/openapiservicetofunctions") | Transforms OpenAPI service specifications into a format compatible with OpenAI's function calling mechanism.  |
| [OutputAdapter](https://docs.haystack.deepset.ai/docs/outputadapter "https://docs.haystack.deepset.ai/docs/outputadapter")                                     | Helps the output of one component fit into the input of another.                                              |
| [PDFMinerToDocument](https://docs.haystack.deepset.ai/docs/pdfminertodocument "https://docs.haystack.deepset.ai/docs/pdfminertodocument")                      | Converts complex PDF files to documents using pdfminer arguments.                                             |
| [PPTXToDocument](https://docs.haystack.deepset.ai/docs/pptxtodocument "https://docs.haystack.deepset.ai/docs/pptxtodocument")                                  | Converts PPTX files to documents.                                                                             |
| [PyPDFToDocument](https://docs.haystack.deepset.ai/docs/pypdftodocument "https://docs.haystack.deepset.ai/docs/pypdftodocument")                               | Converts PDF files to documents.                                                                              |
| [TikaDocumentConverter](https://docs.haystack.deepset.ai/docs/tikadocumentconverter "https://docs.haystack.deepset.ai/docs/tikadocumentconverter")             | Converts various file types to documents using Apache Tika.                                                   |
| [TextFileToDocument](https://docs.haystack.deepset.ai/docs/textfiletodocument "https://docs.haystack.deepset.ai/docs/textfiletodocument")                      | Converts text files to documents.                                                                             |
| [UnstructuredFileConverter](https://docs.haystack.deepset.ai/docs/unstructuredfileconverter "https://docs.haystack.deepset.ai/docs/unstructuredfileconverter") | Converts text files and directories to a document.                                                            |
'''

from typing import Union
from pathlib import Path

from haystack.components.converters import (
    TextFileToDocument,
    PyPDFToDocument,
    DOCXToDocument,
    HTMLToDocument,
    MarkdownToDocument,
    JSONConverter,
    TikaDocumentConverter,
    CSVToDocument,
    PPTXToDocument,
    TextFileToDocument,
)

from haystack.components.converters import AzureOCRDocumentConverter
from haystack_integrations.components.converters.unstructured import UnstructuredFileConverter


class ConverterFactory:
    """Factory class to create appropriate document converters based on file type."""

    def __init__(self):
        self.converter_manager = ConverterManager()
        self._register_default_converters()

    def _register_default_converters(self):
        """Register default converters."""
        self.converter_manager.register_converter(".txt", TextFileToDocument)
        self.converter_manager.register_converter(".pdf", PyPDFToDocument)
        self.converter_manager.register_converter(".docx", DOCXToDocument)
        self.converter_manager.register_converter(".html", HTMLToDocument)
        self.converter_manager.register_converter(".htm", HTMLToDocument)
        self.converter_manager.register_converter(".md", MarkdownToDocument)
        self.converter_manager.register_converter(".json", JSONConverter)
        self.converter_manager.register_converter(".csv", CSVToDocument)
        self.converter_manager.register_converter(".pptx", PPTXToDocument)

    def get_converter(self, file_path: Union[str, Path], provider="local", **kwargs):
        """
        Get appropriate converter based on file extension.

        Args:
            file_path: Path to the file that needs to be converted
            **kwargs: Additional arguments to pass to the converter

        Returns:
            Appropriate converter instance

        Raises:
            ValueError: If file type is not supported
        """
        if provider == "local":
            if isinstance(file_path, str):
                file_path = Path(file_path)

            file_extension = file_path.suffix.lower()

            # Get converter class from manager
            converter_class = self.converter_manager.get_converter(
                file_extension)

            # If no specific converter found, fallback to Tika
            if converter_class is None:
                return TikaDocumentConverter(**kwargs)

            return converter_class(**kwargs)
        elif provider == "azure":
            return AzureOCRDocumentConverter(**kwargs)
        elif provider == "unstructured":
            return UnstructuredFileConverter(**kwargs)

# converter_manager.py


class ConverterManager:
    """Manages registration and retrieval of document converters."""

    def __init__(self):
        self._converters = {}

    def register_converter(self, extension: str, converter_class):
        """Register a converter class for a specific file extension."""
        self._converters[extension.lower()] = converter_class

    def get_converter(self, extension: str):
        """Retrieve a converter class for a specific file extension."""
        return self._converters.get(extension.lower())
