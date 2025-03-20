from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import os
import logging

from haystack import component
from haystack.dataclasses import Document

from rag.image_processor import ImageProcessorManager

logger = logging.getLogger(__name__)


@component
class ImageToDocument:
    """
    将图像文件转换为文档对象。

    支持多种图像格式（jpg、png、gif等）并可以提取图像中的文本（OCR）
    以及分析图像内容（图像描述、标签等）。
    """

    def __init__(self, use_ocr: bool = True, use_image_analysis: bool = True):
        """
        初始化图像转换器。

        Args:
            use_ocr: 是否使用OCR提取图像中的文本
            use_image_analysis: 是否分析图像内容
        """
        self.use_ocr = use_ocr
        self.use_image_analysis = use_image_analysis
        self.image_processor = ImageProcessorManager().get_processor()

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path]]) -> Dict[str, List[Document]]:
        """
        将图像转换为文档。

        Args:
            sources: 图像文件路径列表

        Returns:
            包含转换后的文档对象的字典
        """
        documents = []

        for source in sources:
            try:
                source_path = Path(source)
                if not source_path.exists():
                    logger.warning(f"文件 {source} 不存在，已跳过")
                    continue

                # 将图像转换为文档
                document = self.image_processor.image_to_document(source_path)
                documents.append(document)
                logger.info(f"成功转换图像: {source}")

            except Exception as e:
                logger.error(f"转换图像 {source} 时出错: {str(e)}")

        return {"documents": documents}


@component
class ImageBatchConverter:
    """
    批量处理图像文件夹，并将所有图像转换为文档对象。

    可用于处理包含多个图像的文件夹，如图像数据集或照片集合。
    """

    def __init__(self, recursive: bool = True, extensions: Optional[List[str]] = None):
        """
        初始化批量图像转换器。

        Args:
            recursive: 是否递归扫描子文件夹
            extensions: 要处理的图像文件扩展名列表
        """
        self.recursive = recursive
        self.extensions = extensions or [
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        self.converter = ImageToDocument()

    @component.output_types(documents=List[Document])
    def run(self, directory: Union[str, Path]) -> Dict[str, List[Document]]:
        """
        扫描目录并转换所有图像文件。

        Args:
            directory: 包含图像的目录路径

        Returns:
            包含转换后的文档对象的字典
        """
        directory_path = Path(directory)
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"目录 {directory} 不存在或不是一个有效的目录")

        image_files = []

        # 扫描目录
        if self.recursive:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.suffix.lower() in self.extensions:
                        image_files.append(str(file_path))
        else:
            for file in directory_path.iterdir():
                if file.is_file() and file.suffix.lower() in self.extensions:
                    image_files.append(str(file))

        # 转换所有图像
        if image_files:
            return self.converter.run(image_files)
        else:
            logger.warning(f"目录 {directory} 中没有找到支持的图像文件")
            return {"documents": []}
