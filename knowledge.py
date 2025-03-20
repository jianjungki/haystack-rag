from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from pathlib import Path
import mimetypes
import os
import magic
import nltk
import chainlit as cl
import pandas as pd
import numpy as np
from datetime import datetime

from rag.converter import ConverterManager
from rag.retriever import RetrieverManager
from rag.store import StoreManager
from rag.generator import GeneratorManager, GeneratorConfig
from rag.embedder import EmbedderManager
from rag.evaluator import EvaluatorManager

# Download NLTK data for text processing
nltk.download('punkt_tab')

# Configuration for optimal embedding models by content type
EMBEDDING_MODELS = {
    "default": "malenia1/ternary-weight-embedding",
    "text": "malenia1/ternary-weight-embedding",
    "code": "BAAI/bge-small-en-v1.5",
    "pdf": "sentence-transformers/all-MiniLM-L6-v2",
    "technical": "sentence-transformers/all-mpnet-base-v2",
    "office": "sentence-transformers/all-mpnet-base-v2",
    "image": "clip-ViT-B-32",
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2"
}

# Configuration for document splitters by content type
SPLITTER_CONFIG = {
    "default": {"split_by": "sentence", "split_length": 3},
    "code": {"split_by": "word", "split_length": 100},
    "pdf": {"split_by": "passage", "split_length": 5},
    "technical": {"split_by": "sentence", "split_length": 2},
    "office": {"split_by": "passage", "split_length": 2},
    "image": {"split_by": "sentence", "split_length": 1},
}


class ContentAnalyzer:
    """Analyzes content to determine optimal processing strategy"""

    # 文档格式分类
    DOCUMENT_EXTENSIONS = {
        "pdf": [".pdf"],
        "office": [".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods", ".odp"],
        "text": [".txt", ".md", ".rst", ".text"],
        "code": [".py", ".js", ".java", ".cpp", ".rb", ".php", ".go", ".rs", ".ts", ".cs", ".swift", ".kt"],
        "technical": [".csv", ".xml", ".json", ".yaml", ".yml", ".ini", ".config"]
    }

    # 图片格式分类
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png",
                        ".gif", ".bmp", ".tiff", ".webp", ".svg"]

    @staticmethod
    def detect_file_type(file_path):
        """Detect file type using mime and magic"""
        mime_type, _ = mimetypes.guess_type(file_path)

        # Use python-magic for more accurate detection
        try:
            mime_from_magic = magic.from_file(file_path, mime=True)
            if mime_from_magic:
                mime_type = mime_from_magic
        except:
            pass

        return mime_type

    @staticmethod
    def detect_content_type(file_path):
        """Analyze content to determine optimal processing strategy"""
        mime_type = ContentAnalyzer.detect_file_type(file_path)
        file_ext = Path(file_path).suffix.lower()

        # 检查是否为图片文件
        if file_ext in ContentAnalyzer.IMAGE_EXTENSIONS or (mime_type and 'image/' in mime_type):
            return "image"

        # 根据MIME类型判断
        if mime_type:
            if 'pdf' in mime_type:
                return "pdf"
            elif any(x in mime_type for x in ['javascript', 'python', 'java', 'c++', 'ruby', 'php']):
                return "code"
            elif any(x in mime_type for x in ['msword', 'officedocument', 'spreadsheet', 'presentation']):
                return "office"
            elif any(x in mime_type for x in ['csv', 'excel', 'xml']):
                return "technical"

        # 根据文件扩展名判断
        for content_type, extensions in ContentAnalyzer.DOCUMENT_EXTENSIONS.items():
            if file_ext in extensions:
                return content_type

        # 默认为文本类型
        return "text"

    @staticmethod
    def get_optimal_embedder(content_type):
        """Get optimal embedding model for content type"""
        return EMBEDDING_MODELS.get(content_type, EMBEDDING_MODELS["default"])

    @staticmethod
    def get_optimal_splitter_config(content_type):
        """Get optimal document splitter config for content type"""
        return SPLITTER_CONFIG.get(content_type, SPLITTER_CONFIG["default"])

    @staticmethod
    def get_optimal_converter(file_path):
        """Get optimal converter based on file analysis"""
        mime_type = ContentAnalyzer.detect_file_type(file_path)
        file_ext = Path(file_path).suffix.lower()

        # 图片处理
        if file_ext in ContentAnalyzer.IMAGE_EXTENSIONS or (mime_type and 'image/' in mime_type):
            # 使用OCR处理图片
            try:
                return ConverterManager().get_converter(file_path, provider="azure")
            except:
                try:
                    return ConverterManager().get_converter(file_path, provider="unstructured")
                except:
                    # 如果OCR不可用，尝试其他方法
                    pass

        # PDF处理
        if file_ext == '.pdf' or (mime_type and 'pdf' in mime_type):
            # 先尝试检测是否为扫描PDF
            try:
                # 如果是图像PDF，使用OCR
                if 'image' in mime_type:
                    return ConverterManager().get_converter(file_path, provider="azure")
                # 普通PDF
                return ConverterManager().get_converter(file_path, provider="unstructured")
            except:
                return ConverterManager().get_converter(file_path)

        # Office文档处理
        if file_ext in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
            try:
                return ConverterManager().get_converter(file_path, provider="unstructured")
            except:
                return ConverterManager().get_converter(file_path)

        # 默认使用标准转换器
        try:
            return ConverterManager().get_converter(file_path, provider="unstructured")
        except:
            # 回退到基础转换器
            return ConverterManager().get_converter(file_path)


@cl.on_chat_start
async def on_chat_start():
    app_user = cl.user_session.get("user")
    await cl.Message(f"Hello {app_user.identifier}. Welcome to the intelligent knowledge base.").send()

    # 显示使用指南
    usage_guide = """
    ## 智能知识库使用指南
    
    1. **上传文件**: 支持多种格式，包括PDF、Office文档、图片等
    2. **提问问题**: 系统会分析文档内容并给出回答
    3. **查看评估**: 每次回答后会显示RAG质量评估结果
    4. **评估指标**:
       - 忠实度: 回答内容是否忠实于检索到的文档
       - 上下文相关度: 检索到的文档与问题的相关程度
       - 回答相关度: 回答与问题的相关程度
       - 检索文档数: 用于生成回答的文档数量
    5. **特殊命令**: 
       - `/eval`: 显示完整评估历史和平均分
    """

    await cl.Message(content=usage_guide).send()

    files = None

    # 显示支持的文件类型
    converter_manager = ConverterManager()
    supported_types = converter_manager.get_supported_file_types()

    file_types_msg = "支持的文件类型:\n"
    for category, extensions in supported_types.items():
        file_types_msg += f"- {category.capitalize()}: {', '.join(extensions)}\n"

    await cl.Message(content=file_types_msg).send()

    # 等待用户上传文件
    while files == None:
        files = await cl.AskFileMessage(
            content="请上传任意文件（PDF、文档、表格、演示文稿、图像等）开始！",
            accept=["*/*"],  # 接受所有文件类型
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"处理文件 `{file.name}`...")
    await msg.send()

    # 初始化文档存储
    document_store = StoreManager().get_store("memory")

    # 分析内容以确定最佳处理策略
    content_type = ContentAnalyzer.detect_content_type(file.path)
    await cl.Message(content=f"检测到内容类型: `{content_type}`").send()

    # 基于内容类型获取最佳嵌入模型
    embedding_model = ContentAnalyzer.get_optimal_embedder(content_type)

    # 获取最佳文档嵌入器
    embedder = EmbedderManager().get_embedder(
        provider="sentence_transformer",
        embedder_type="document",
        embedding_model=embedding_model
    )

    # 基于文件分析获取最佳转换器
    converter = ContentAnalyzer.get_optimal_converter(file.path)

    # 获取最佳分割器配置
    splitter_config = ContentAnalyzer.get_optimal_splitter_config(content_type)

    # 建立索引流程
    indexing = Pipeline()
    indexing.add_component("converter", converter)
    indexing.add_component("cleaner", DocumentCleaner())
    indexing.add_component("splitter", DocumentSplitter(**splitter_config))
    indexing.add_component("doc_embedder", embedder)
    indexing.add_component("writer", DocumentWriter(document_store))

    indexing.connect("converter", "cleaner")
    indexing.connect("cleaner", "splitter")
    indexing.connect("splitter", "doc_embedder")
    indexing.connect("doc_embedder", "writer")

    # 运行索引
    await cl.Message(content=f"正在为 {content_type} 内容使用最佳设置索引 `{file.name}`...").send()
    print(
        f"索引 `{file.path}` 作为 {content_type} 内容，使用 {embedding_model} 嵌入模型...")
    indexing.run({"converter": {"sources": [file.path]}})

    # 设置语言模型和RAG流程
    llm_config = GeneratorConfig(
        api_base_url="https://api.siliconflow.cn/v1",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        api_key="sk-aqilqiwnnwgdbzvqyriehstlzluawydzuimomxwvmlqfgfzk",
        temperature=0.5,
        max_tokens=4096
    )
    generator = GeneratorManager().get_generator("openai", False, llm_config)

    # 设置带有内容特定提示的RAG组件
    if content_type == "image":
        template = """
            请仅基于以下从图像中提取的信息回答问题。不要使用任何其他知识，如果无法从提供的信息中找到答案，请直接说明无法回答。

            图像信息:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            问题: {{question}}
            回答:
        """
    elif content_type == "code":
        template = """
            请仅基于以下代码文档信息回答技术问题。严格限制在提供的代码内容范围内回答，不要添加任何未在上下文中出现的信息或解释。如果无法从代码中找到答案，请直接说明。

            代码文档:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            问题: {{question}}
            回答:
        """
    elif content_type == "pdf" or content_type == "office":
        template = """
            请仅基于以下文档内容回答问题，不要添加任何额外信息。如果文档中没有提供足够信息回答问题，请明确指出"文档中没有提供相关信息"，而不是猜测或使用外部知识。

            文档内容:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            问题: {{question}}
            回答:
        """
    else:
        template = """
            请仅使用以下{content_type}文档中提供的信息回答问题。不要添加任何文档中未包含的信息，不要使用自己的知识，不要进行推测。如果文档信息不足以回答问题，请直接说明"提供的文档中没有这方面的信息"。

            上下文:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            问题: {{question}}
            回答:
        """.format(content_type=content_type)

    prompt_builder = PromptBuilder(template=template)

    # 设置与文档嵌入器使用相同嵌入模型的检索器
    retriever = RetrieverManager().get_retriever(
        "embedding",
        document_store=document_store,
        embedding_model=embedding_model
    )

    # 文本嵌入器也使用相同的模型以保持一致性
    text_embedder = EmbedderManager().get_embedder(
        "sentence_transformer",
        "text",
        embedding_model=embedding_model
    )

    # 设置RAG流程
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", generator)

    # 设置评估器
    evaluator_manager = EvaluatorManager()
    faithfulness_evaluator = evaluator_manager.get_evaluator("faithfulness")
    context_relevancy_evaluator = evaluator_manager.get_evaluator(
        "context_relevancy")
    answer_relevancy_evaluator = evaluator_manager.get_evaluator(
        "answer_relevancy")

    # 设置评估流程
    eval_pipeline = Pipeline()
    eval_pipeline.add_component("faithfulness", faithfulness_evaluator)
    eval_pipeline.add_component(
        "context_relevancy", context_relevancy_evaluator)
    eval_pipeline.add_component("answer_relevancy", answer_relevancy_evaluator)

    # 连接组件
    rag_pipeline.connect("text_embedder.embedding",
                         "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    # 在会话中存储流程和内容信息
    cl.user_session.set("chain", rag_pipeline)
    cl.user_session.set("content_type", content_type)
    cl.user_session.set("embedding_model", embedding_model)
    cl.user_session.set("file_name", file.name)
    cl.user_session.set("evaluators", {
        "faithfulness": faithfulness_evaluator,
        "context_relevancy": context_relevancy_evaluator,
        "answer_relevancy": answer_relevancy_evaluator
    })

    # 初始化评估历史记录
    cl.user_session.set("evaluation_history", [])

    await cl.Message(
        content=f"准备就绪！现在您可以提问关于 `{file.name}` 的问题。\n使用的嵌入模型: {embedding_model}").send()


@cl.on_message
async def main(message: cl.Message):
    # 检查是否为特殊命令
    if message.content.strip().lower() == "/eval":
        # 显示完整评估历史
        evaluation_history = cl.user_session.get("evaluation_history", [])
        if not evaluation_history:
            await cl.Message(content="还没有评估历史记录。请先提问一些问题。").send()
            return

        # 创建完整历史记录表格
        history_data = []
        for i, item in enumerate(evaluation_history):
            history_data.append([
                i+1,
                item["timestamp"],
                item["question"][:50] +
                "..." if len(item["question"]) > 50 else item["question"],
                item["metrics"].get("忠实度", "N/A"),
                item["metrics"].get("上下文相关度", "N/A"),
                item["metrics"].get("回答相关度", "N/A"),
                item["metrics"].get("检索文档数", 0)
            ])

        # 创建表格
        columns = ["序号", "时间", "问题", "忠实度", "上下文相关度", "回答相关度", "检索文档数"]
        history_table = cl.Table(data=history_data, columns=columns)

        await cl.Message(content="完整评估历史记录:", elements=[history_table]).send()

        # 计算平均评分
        metrics = ["忠实度", "上下文相关度", "回答相关度", "检索文档数"]
        avg_scores = {}

        for metric in metrics:
            values = [h["metrics"].get(
                metric, 0) for h in evaluation_history if metric in h["metrics"]]
            if values:
                avg_scores[metric] = sum(values) / len(values)
            else:
                avg_scores[metric] = "N/A"

        # 创建平均分表格
        avg_data = [[metric, avg_scores[metric]] for metric in metrics]
        avg_table = cl.Table(data=avg_data, columns=["评估指标", "平均分"])

        await cl.Message(content="评估指标平均分:", elements=[avg_table]).send()
        return

    chain = cl.user_session.get("chain")
    content_type = cl.user_session.get("content_type", "text")
    file_name = cl.user_session.get("file_name", "文档")

    # 获取评估器
    evaluator_manager = EvaluatorManager()
    faithfulness_evaluator = evaluator_manager.get_evaluator("faithfulness")
    context_relevancy_evaluator = evaluator_manager.get_evaluator(
        "context_relevancy")
    answer_relevancy_evaluator = evaluator_manager.get_evaluator(
        "answer_relevancy")

    evaluation_history = cl.user_session.get("evaluation_history", [])

    # 显示思考状态
    thinking_msg = cl.Message(content="正在分析文档内容并生成回答...")
    await thinking_msg.send()

    try:
        # 运行RAG流程
        res = chain.run(
            {
                "text_embedder": {"text": message.content},
                "prompt_builder": {
                    "question": message.content,
                    "content_type": content_type
                }
            }
        )

        answer = res["llm"]["replies"][0]
        retrieved_docs = res.get("retriever", {}).get("documents", [])

        # 更新思考消息
        thinking_msg.content = answer
        await thinking_msg.update()

        # 执行评估
        eval_results = {}

        # 评估忠实度
        try:
            faithfulness_score = faithfulness_evaluator.run(
                question=message.content,
                answer=answer,
                documents=retrieved_docs
            )
            eval_results["忠实度"] = faithfulness_score["score"]
        except Exception as eval_err:
            print(f"忠实度评估错误: {str(eval_err)}")

        # 评估上下文相关度
        try:
            context_score = context_relevancy_evaluator.run(
                question=message.content,
                documents=retrieved_docs
            )
            eval_results["上下文相关度"] = context_score["score"]
        except Exception as eval_err:
            print(f"上下文相关度评估错误: {str(eval_err)}")

        # 评估回答相关度
        try:
            answer_score = answer_relevancy_evaluator.run(
                question=message.content,
                answer=answer
            )
            eval_results["回答相关度"] = answer_score["score"]
        except Exception as eval_err:
            print(f"回答相关度评估错误: {str(eval_err)}")

        # 计算检索文档数量
        doc_count = len(retrieved_docs) if retrieved_docs else 0
        eval_results["检索文档数"] = doc_count

        # 记录评估结果
        evaluation_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": message.content,
            "answer": answer,
            "metrics": eval_results
        })

        # 更新会话中的评估历史
        cl.user_session.set("evaluation_history", evaluation_history)

        # 创建评估结果表格
        if eval_results:
            # 创建表格数据
            table_data = {
                "评估指标": list(eval_results.keys()),
                "得分": list(eval_results.values())
            }

            # 创建Pandas DataFrame
            df = pd.DataFrame(table_data)

            # 创建可视化表格元素
            table = cl.Table(
                data=df.values.tolist(),
                columns=df.columns.tolist()
            )

            # 发送评估结果表格
            await cl.Message(content="RAG检索与回答质量评估结果:", elements=[table]).send()

            # 如果有足够的历史数据，显示趋势图
            if len(evaluation_history) >= 3:
                # 准备历史趋势数据
                history_df = pd.DataFrame([
                    {**{"问题序号": i+1}, **item["metrics"]}
                    # 最多显示最近10次
                    for i, item in enumerate(evaluation_history[-10:])
                ])

                # 创建趋势表格
                trend_table = cl.Table(
                    data=history_df.values.tolist(),
                    columns=history_df.columns.tolist()
                )

                await cl.Message(content="历史评估趋势:", elements=[trend_table]).send()

            # 提示用户可以使用/eval命令
            if len(evaluation_history) >= 5:  # 有一定历史记录后提示
                await cl.Message(content="提示: 输入 `/eval` 可查看完整评估历史和平均分").send()

    except Exception as e:
        # 异常处理
        error_message = f"处理您的请求时出错: {str(e)}"
        thinking_msg.content = error_message
        await thinking_msg.update()
        print(f"错误: {str(e)}")


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # 从数据库获取匹配的用户并将哈希密码与数据库中存储的值进行比较
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
