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

from typing import Union, Dict, List
from pathlib import Path
import mimetypes
import magic
import logging

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
    PDFMinerToDocument,
)

from haystack.components.converters import AzureOCRDocumentConverter
from haystack_integrations.components.converters.unstructured import UnstructuredFileConverter

# 导入自定义图像转换器
try:
    from rag.image_converter import ImageToDocument
    HAS_IMAGE_CONVERTER = True
except ImportError:
    HAS_IMAGE_CONVERTER = False

logger = logging.getLogger(__name__)


class ConverterManager:
    """Manages registration and retrieval of document converters with singleton pattern."""
    _instance = None

    # 支持的文件类型和扩展名的映射
    FILE_TYPE_EXTENSIONS = {
        "text": [".txt", ".text"],
        "pdf": [".pdf"],
        "document": [".doc", ".docx", ".odt"],
        "spreadsheet": [".xls", ".xlsx", ".ods", ".csv"],
        "presentation": [".ppt", ".pptx", ".odp"],
        "html": [".html", ".htm"],
        "markdown": [".md", ".markdown"],
        "json": [".json"],
        "xml": [".xml"],
        "code": [".py", ".js", ".java", ".cpp", ".cs", ".php", ".rb", ".go", ".rs", ".ts"],
        "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".svg"],
    }

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
                # 基本转换器
                "local.txt": TextFileToDocument,
                "local.pdf": PyPDFToDocument,
                "local.pdf_miner": PDFMinerToDocument,  # 高级PDF处理
                "local.docx": DOCXToDocument,
                "local.html": HTMLToDocument,
                "local.htm": HTMLToDocument,
                "local.md": MarkdownToDocument,
                "local.json": JSONConverter,
                "local.csv": CSVToDocument,
                "local.pptx": PPTXToDocument,
                "local.tika": TikaDocumentConverter,
                # 特定提供商的转换器
                "azure": AzureOCRDocumentConverter,
                "unstructured": UnstructuredFileConverter,
            }

            # 如果图像转换器可用，添加它
            if HAS_IMAGE_CONVERTER:
                self._converters["image"] = ImageToDocument

            self._initialized = True

    def register_converter(self, key: str, converter_class) -> None:
        """Register a converter class for a specific key."""
        self._converters[key.lower()] = converter_class

    def unregister_converter(self, key: str) -> None:
        """Remove a converter from the registry."""
        if key.lower() in self._converters:
            del self._converters[key.lower()]

    def detect_file_type(self, file_path: Union[str, Path]) -> str:
        """
        检测文件类型，返回适当的处理器类型。

        Args:
            file_path: 文件路径

        Returns:
            文件类型标识符
        """
        file_path = Path(file_path) if isinstance(
            file_path, str) else file_path
        file_extension = file_path.suffix.lower()

        # 使用MIME类型和magic进行更准确的检测
        mime_type = None
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            # 尝试使用python-magic获取更准确的MIME类型
            try:
                mime_from_magic = magic.from_file(str(file_path), mime=True)
                if mime_from_magic:
                    mime_type = mime_from_magic
            except:
                pass
        except:
            pass

        # 基于MIME类型检测
        if mime_type:
            if 'image/' in mime_type:
                return "image"
            elif 'pdf' in mime_type:
                # 检查是否为扫描PDF，这可能需要OCR
                try:
                    if 'image' in mime_type or 'scan' in mime_type:
                        return "azure"  # 使用OCR处理扫描的PDF
                except:
                    pass
                return "local.pdf"
            elif 'msword' in mime_type or 'officedocument.wordprocessingml' in mime_type:
                return "local.docx"
            elif 'officedocument.spreadsheetml' in mime_type or 'csv' in mime_type:
                return "local.csv"
            elif 'officedocument.presentationml' in mime_type:
                return "local.pptx"
            elif 'html' in mime_type:
                return "local.html"
            elif 'text/plain' in mime_type:
                return "local.txt"
            elif 'text/markdown' in mime_type:
                return "local.md"
            elif 'application/json' in mime_type:
                return "local.json"

        # 基于文件扩展名检测
        for file_type, extensions in self.FILE_TYPE_EXTENSIONS.items():
            if file_extension in extensions:
                if file_type == "image":
                    return "image"
                elif file_type == "pdf":
                    return "local.pdf"
                elif file_type == "document":
                    return "local.docx"
                elif file_type == "spreadsheet":
                    return "local.csv"
                elif file_type == "presentation":
                    return "local.pptx"
                elif file_type == "html":
                    return "local.html"
                elif file_type == "markdown":
                    return "local.md"
                elif file_type == "json":
                    return "local.json"
                elif file_type == "text":
                    return "local.txt"

        # 如果无法确定，尝试使用Tika或者Unstructured
        try:
            # 先尝试不那么重的Tika
            return "local.tika"
        except:
            # 然后尝试Unstructured
            return "unstructured"

    def get_converter(self, file_path: Union[str, Path], provider: str = "auto", **kwargs):
        """
        Get appropriate converter based on file extension and provider.

        Args:
            file_path: Path to the file that needs to be converted
            provider: Provider to use for conversion ("auto", "local", "azure", "unstructured", "image")
            **kwargs: Additional arguments to pass to the converter

        Returns:
            Appropriate converter instance

        Raises:
            ValueError: If provider or file type is not supported
        """
        if provider == "auto":
            # 自动检测最适合的转换器
            detected_type = self.detect_file_type(file_path)

            # 检查是否为图像类型
            if detected_type == "image" and HAS_IMAGE_CONVERTER:
                return self._converters["image"](**kwargs)

            # 检查是否需要特殊处理器
            if detected_type in ["azure", "unstructured"]:
                return self._converters[detected_type](**kwargs)

            # 否则使用本地转换器
            if detected_type.startswith("local."):
                converter_key = detected_type
                converter_class = self._converters.get(converter_key)

                if converter_class:
                    return converter_class(**kwargs)

            # 如果没有合适的转换器，尝试Unstructured
            try:
                return self._converters["unstructured"](**kwargs)
            except:
                # 最后尝试TikaDocumentConverter
                return TikaDocumentConverter(**kwargs)

        elif provider == "local":
            if isinstance(file_path, str):
                file_path = Path(file_path)

            file_extension = file_path.suffix.lower()
            converter_key = f"local{file_extension}"

            converter_class = self._converters.get(converter_key)
            if converter_class is None:
                return TikaDocumentConverter(**kwargs)

            return converter_class(**kwargs)

        elif provider == "image" and HAS_IMAGE_CONVERTER:
            return self._converters["image"](**kwargs)

        elif provider in self._converters:
            return self._converters[provider](**kwargs)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def list_available_converters(self) -> list:
        """List all registered converter types."""
        return list(self._converters.keys())

    def get_supported_file_types(self) -> Dict[str, List[str]]:
        """
        获取支持的文件类型和对应的扩展名

        Returns:
            文件类型及其扩展名的字典
        """
        return self.FILE_TYPE_EXTENSIONS
