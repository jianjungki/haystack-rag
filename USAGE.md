# 智能知识库系统使用说明

本文档介绍如何使用智能知识库系统，该系统能够自动识别和处理多种文件格式，并使用最适合的方式提取、分析和检索信息。

## 支持的文件格式

系统支持多种文件格式，包括但不限于：

### 文档格式
- PDF文件 (`.pdf`)
- Microsoft Word文档 (`.doc`, `.docx`)
- Microsoft Excel电子表格 (`.xls`, `.xlsx`)
- Microsoft PowerPoint演示文稿 (`.ppt`, `.pptx`)
- 文本文件 (`.txt`)
- HTML文件 (`.html`, `.htm`)
- Markdown文件 (`.md`)
- JSON文件 (`.json`)
- CSV文件 (`.csv`)
- XML文件 (`.xml`)

### 图片格式
- JPEG图像 (`.jpg`, `.jpeg`)
- PNG图像 (`.png`)
- GIF图像 (`.gif`)
- BMP图像 (`.bmp`)
- TIFF图像 (`.tiff`)
- WebP图像 (`.webp`)
- SVG图像 (`.svg`)

### 代码文件
- Python (`.py`)
- JavaScript (`.js`)
- Java (`.java`)
- C++ (`.cpp`)
- C# (`.cs`)
- PHP (`.php`)
- Ruby (`.rb`)
- Go (`.go`)
- Rust (`.rs`)
- TypeScript (`.ts`)

## 安装和配置

### 1. 安装依赖

使用以下命令安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥（可选）

如果您想使用Azure OCR功能进行图像处理，设置以下环境变量：

```bash
export AZURE_VISION_KEY=your_azure_key
export AZURE_VISION_ENDPOINT=your_azure_endpoint
```

如果您想使用自己的OpenAI API密钥，可以在`knowlage.py`文件中修改，或设置环境变量：

```bash
export OPENAI_API_KEY=your_openai_key
```

### 3. 安装Tesseract OCR（可选，推荐用于图像处理）

对于图像文件中的文本识别，建议安装Tesseract OCR：

- Windows: 下载并安装[Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- MacOS: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

## 运行系统

使用以下命令启动知识库系统：

```bash
chainlit run knowlage.py
```

此命令将启动web服务器，通常在http://localhost:8000上可访问。

## 使用流程

### 1. 登录系统

使用以下凭据登录：
- 用户名: `admin`
- 密码: `admin`

### 2. 上传文件

1. 系统启动后会显示支持的文件类型列表
2. 点击"上传文件"按钮，选择您要分析的文件
3. 系统会自动检测文件类型并选择最佳处理策略
4. 文件处理过程中会显示进度信息

### 3. 提问

一旦文件被索引，您就可以开始提问了：

1. 在聊天框中输入您的问题
2. 系统会自动分析文档内容并生成答案
3. 回答将基于文档内容，使用针对特定内容类型的最佳提示模板

### 4. 文件类型特殊处理

系统会根据不同的文件类型采用不同的处理策略：

- **图像文件**：使用OCR提取文本，并分析图像内容（标签、描述等）
- **PDF文件**：自动检测是否为扫描PDF，并使用适当的方法（如OCR）处理
- **Office文档**：智能提取结构化内容
- **代码文件**：使用特殊的代码理解模型和分割策略
- **技术文档**：采用适合技术内容的处理方式

## 扩展功能

### 批量处理图像

如果您需要处理大量图像，可以使用`ImageBatchConverter`组件：

```python
from rag.image_converter import ImageBatchConverter

# 创建批量转换器
converter = ImageBatchConverter(recursive=True)

# 处理整个图像目录
results = converter.run("/path/to/images")
```

### 自定义嵌入模型

如果您想为特定内容类型使用自定义嵌入模型，可以修改`EMBEDDING_MODELS`字典：

```python
# 在knowlage.py中
EMBEDDING_MODELS = {
    "default": "your-default-model",
    "text": "your-text-model",
    # 添加更多自定义模型...
}
```

## 故障排除

### 文件无法识别

如果系统无法正确识别文件类型：

1. 检查文件扩展名是否正确
2. 确保文件没有损坏
3. 尝试将文件转换为更标准的格式

### OCR质量不佳

如果图像或扫描PDF的OCR结果质量不佳：

1. 确保图像清晰度足够高
2. 检查是否安装了Tesseract OCR
3. 考虑使用Azure Vision API（需要配置API密钥）

### 内存问题

处理大型文件时可能遇到内存问题：

1. 增加系统可用内存
2. 尝试将大文件拆分为多个小文件
3. 调整`splitter_config`中的参数以减小处理单元

## 联系与支持

如有问题或建议，请联系系统管理员或提交GitHub Issue。 