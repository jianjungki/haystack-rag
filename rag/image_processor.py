import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pathlib import Path
from typing import List, Dict, Union, Optional
from haystack.dataclasses import Document

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForVision2Seq
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from msrest.authentication import CognitiveServicesCredentials
    HAS_AZURE_VISION = True
except ImportError:
    HAS_AZURE_VISION = False


class ImageProcessor:
    """处理图像内容的工具类"""

    def __init__(self):
        """初始化图像处理器"""
        self.ocr_engine = self._initialize_ocr()
        self.image_model = self._initialize_image_model()
        self.azure_client = self._initialize_azure_vision()

    def _initialize_ocr(self):
        """初始化OCR引擎"""
        try:
            # 检查pytesseract是否正确配置
            pytesseract.get_tesseract_version()
            return "tesseract"
        except:
            return None

    def _initialize_image_model(self):
        """初始化图像模型"""
        if not HAS_TRANSFORMERS:
            return None

        try:
            # 尝试加载CLIP模型
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32")
            return {"model": model, "processor": processor, "type": "clip"}
        except:
            try:
                # 尝试加载其他Vision模型
                model = AutoModelForVision2Seq.from_pretrained(
                    "Salesforce/blip-image-captioning-base")
                processor = AutoProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base")
                return {"model": model, "processor": processor, "type": "blip"}
            except:
                return None

    def _initialize_azure_vision(self):
        """初始化Azure Vision API"""
        if not HAS_AZURE_VISION:
            return None

        try:
            # 从环境变量获取Azure密钥
            subscription_key = os.environ.get("AZURE_VISION_KEY")
            endpoint = os.environ.get("AZURE_VISION_ENDPOINT")

            if subscription_key and endpoint:
                return ComputerVisionClient(
                    endpoint=endpoint,
                    credentials=CognitiveServicesCredentials(subscription_key)
                )
            return None
        except:
            return None

    def extract_text_from_image(self, image_path: Union[str, Path]) -> str:
        """从图像中提取文本"""
        image_path = str(image_path)

        # 1. 尝试使用Azure Vision API
        if self.azure_client:
            try:
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                    result = self.azure_client.recognize_printed_text_in_stream(
                        image_data)

                    text = ""
                    for region in result.regions:
                        for line in region.lines:
                            for word in line.words:
                                text += word.text + " "
                            text += "\n"

                    if text.strip():
                        return text
            except:
                pass

        # 2. 尝试使用本地Tesseract OCR
        if self.ocr_engine == "tesseract":
            try:
                text = pytesseract.image_to_string(Image.open(image_path))
                if text.strip():
                    return text
            except:
                pass

        # 3. 简单的图像预处理后再尝试OCR
        try:
            # 读取图像
            image = cv2.imread(image_path)
            # 转换为灰度
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 应用自适应阈值
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            # 执行OCR
            text = pytesseract.image_to_string(thresh)
            if text.strip():
                return text
        except:
            pass

        # 如果所有OCR方法都失败，返回空字符串
        return "无法从图像中提取文本。这可能是一个没有文字的图像。"

    def analyze_image_content(self, image_path: Union[str, Path]) -> Dict:
        """分析图像内容，提取标签、描述和其他信息"""
        image_path = str(image_path)
        result = {"tags": [], "caption": "", "objects": []}

        # 1. 尝试使用Azure Vision API
        if self.azure_client:
            try:
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()

                    # 获取描述
                    description = self.azure_client.describe_image_in_stream(
                        image_data)
                    if description.captions:
                        result["caption"] = description.captions[0].text

                    # 获取标签
                    tags = self.azure_client.tag_image_in_stream(image_data)
                    result["tags"] = [tag.name for tag in tags.tags]

                    # 获取对象
                    objects = self.azure_client.detect_objects_in_stream(
                        image_data)
                    result["objects"] = [
                        obj.object_property for obj in objects.objects]

                    return result
            except:
                pass

        # 2. 尝试使用CLIP或其他模型
        if self.image_model and HAS_TRANSFORMERS:
            try:
                model_info = self.image_model
                model_type = model_info["type"]
                model = model_info["model"]
                processor = model_info["processor"]

                if model_type == "clip":
                    # 预定义一些常见标签进行分类
                    candidate_labels = [
                        "document", "text", "chart", "graph", "table",
                        "landscape", "portrait", "animal", "building", "vehicle",
                        "food", "plant", "indoor scene", "outdoor scene", "screenshot"
                    ]

                    # 处理图像
                    image = Image.open(image_path)
                    inputs = processor(
                        text=candidate_labels,
                        images=image,
                        return_tensors="pt",
                        padding=True
                    )

                    # 获取预测结果
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1)

                    # 获取最相关的标签
                    for i, label in enumerate(candidate_labels):
                        if probs[0][i] > 0.3:  # 只保留概率大于0.3的标签
                            result["tags"].append(label)

                elif model_type == "blip":
                    # 使用BLIP模型生成图像描述
                    image = Image.open(image_path)
                    inputs = processor(images=image, return_tensors="pt")

                    with torch.no_grad():
                        generated_ids = model.generate(
                            pixel_values=inputs.pixel_values, max_length=30)

                    result["caption"] = processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0].strip()

                return result
            except:
                pass

        # 如果所有分析方法都失败，返回基本信息
        try:
            image = Image.open(image_path)
            result["caption"] = f"图像：宽度 {image.width}px，高度 {image.height}px，格式 {image.format}"
            return result
        except:
            return {"tags": [], "caption": "无法分析图像内容", "objects": []}

    def image_to_document(self, image_path: Union[str, Path]) -> Document:
        """将图像转换为文档对象"""
        # 提取文本
        text_content = self.extract_text_from_image(image_path)

        # 分析图像内容
        image_analysis = self.analyze_image_content(image_path)

        # 构建文档内容
        content = []

        if image_analysis["caption"]:
            content.append(f"图像描述: {image_analysis['caption']}")

        if image_analysis["tags"]:
            content.append(f"图像标签: {', '.join(image_analysis['tags'])}")

        if image_analysis["objects"]:
            content.append(f"检测到的对象: {', '.join(image_analysis['objects'])}")

        if text_content:
            content.append(f"图像中的文本:\n{text_content}")

        # 创建文档对象
        return Document(
            content="\n\n".join(content),
            metadata={
                "source": str(image_path),
                "file_type": "image",
                "has_text": bool(text_content.strip()),
                "tags": image_analysis["tags"],
                "caption": image_analysis["caption"]
            }
        )


class ImageProcessorManager:
    """管理图像处理器的工厂类，使用单例模式"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageProcessorManager, cls).__new__(cls)
            cls._instance._processor = None
        return cls._instance

    def get_processor(self) -> ImageProcessor:
        """获取或创建图像处理器实例"""
        if self._processor is None:
            self._processor = ImageProcessor()
        return self._processor
