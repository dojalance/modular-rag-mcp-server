# 模块化企业知识检索与 Agent RAG

基于 MCP 的知识检索服务，面向 PDF 知识库提供混合检索、带引用的片段和摄取及查询 Trace。

## 功能

- PDF 转换、递归分块、元数据提取、向量化和索引写入。
- BM25 与 Dense 双路召回、RRF 融合及可选 Cross-Encoder 精排。
- 通过 stdio 提供 `query_knowledge_hub`、`list_collections` 和 `get_document_summary` 三个工具。
- 对文本层可用的论文支持按页提取并保留页码；对错误号及复合词保留完整 BM25 词项。

## 启动

推荐 Python 3.12。创建并激活虚拟环境后安装依赖，按 `config/settings.yaml` 配置 LLM、Embedding 和存储。模型凭据填入本地配置或使用对应 Provider 的环境变量。

```bash
python -m pip install -e ".[dev]"
python scripts/ingest.py --path path/to/manual.pdf --collection knowledge_hub
python main.py
```

在 MCP Host 中将 `command` 指向该虚拟环境的 Python，启动参数填 `-m src.mcp_server.server`，工作目录设为仓库根目录。先用 `list_collections` 核对集合，再调用 `query_knowledge_hub`。

## 评测

- [PDF 摄取对照与评估口径](docs/evaluation.md)
- [检索与回答评估脚本](scripts/quality_eval/)
