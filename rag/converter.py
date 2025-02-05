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


class ConverterManager:
    """Manages registration and retrieval of document converters with singleton pattern."""
    _instance = None

    def __new__(cls):
        """Singleton pattern to ensure only one manager instance exists."""
        if cls._instance is None:
            cls._instance = super(ConverterManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the converter registry."""
        if not hasattr(self, '_initialized'):
            self._initialized = False
        if not self._initialized:
            self._converters = {
                # Local converters
                "local.txt": TextFileToDocument,
                "local.pdf": PyPDFToDocument,
                "local.docx": DOCXToDocument,
                "local.html": HTMLToDocument,
                "local.htm": HTMLToDocument,
                "local.md": MarkdownToDocument,
                "local.json": JSONConverter,
                "local.csv": CSVToDocument,
                "local.pptx": PPTXToDocument,
                "local.tika": TikaDocumentConverter,
                # Provider-specific converters
                "azure": AzureOCRDocumentConverter,
                "unstructured": UnstructuredFileConverter,
            }
            self._initialized = True

    def register_converter(self, key: str, converter_class) -> None:
        """Register a converter class for a specific key."""
        self._converters[key.lower()] = converter_class

    def unregister_converter(self, key: str) -> None:
        """Remove a converter from the registry."""
        if key.lower() in self._converters:
            del self._converters[key.lower()]

    def get_converter(self, file_path: Union[str, Path], provider: str = "local", **kwargs):
        """
        Get appropriate converter based on file extension and provider.

        Args:
            file_path: Path to the file that needs to be converted
            provider: Provider to use for conversion ("local", "azure", "unstructured")
            **kwargs: Additional arguments to pass to the converter

        Returns:
            Appropriate converter instance

        Raises:
            ValueError: If provider or file type is not supported
        """
        if provider == "local":
            if isinstance(file_path, str):
                file_path = Path(file_path)

            file_extension = file_path.suffix.lower()
            converter_key = f"local{file_extension}"

            converter_class = self._converters.get(converter_key)
            if converter_class is None:
                return TikaDocumentConverter(**kwargs)

            return converter_class(**kwargs)

        elif provider in self._converters:
            return self._converters[provider](**kwargs)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def list_available_converters(self) -> list:
        """List all registered converter types."""
        return list(self._converters.keys())
