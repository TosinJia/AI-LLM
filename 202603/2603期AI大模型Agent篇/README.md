# 虚拟环境
## llamaindex虚拟环境
```
conda env list
conda deactivate
conda remove -n conda_llm-llamaindex --all
conda create -n conda_llm-llamaindex python=3.12
conda activate conda_llm-llamaindex
pip install python-dotenv
pip install llama-index llama-index-embeddings-huggingface llama-index-llms-dashscope llama-index-llms-ollama
pip install llama-index-llms-deepseek
pip install llama-index-utils-workflow

pip install llama-index-readers-file

pip install tree_sitter
pip install tree_sitter_language_pack

pip install llama-index-vector-stores-chroma

pip install llama_index.storage.kvstore.redis

pip install llama-index-storage-docstore-redis
pip install llama-index-vector-stores-redis
pip install llama-index-storage-index-store-redis

pip install 'llama-index-workflows[server]'

pip install llama-index-retrievers-bm25

pip install ragas
pip install rapidfuzz
```

## 问题
### 问题1
- 问题描述
```
F:\iEnviroment\development\python\anaconda3\envs\conda_llm-llamaindex\python.exe F:\PersonalPromotion\AI\code\AI\AI大模型\202603\2603期AI大模型Agent篇\3-llamaindex初始-2026-3-26-初见\01-入门案例.py 
Traceback (most recent call last):
  File "F:\PersonalPromotion\AI\code\AI\AI大模型\202603\2603期AI大模型Agent篇\3-llamaindex初始-2026-3-26-初见\01-入门案例.py", line 1, in <module>
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
  File "F:\iEnviroment\development\python\anaconda3\envs\conda_llm-llamaindex\Lib\site-packages\llama_index\core\__init__.py", line 22, in <module>
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding
  File "F:\iEnviroment\development\python\anaconda3\envs\conda_llm-llamaindex\Lib\site-packages\llama_index\core\embeddings\__init__.py", line 1, in <module>
    from llama_index.core.base.embeddings.base import BaseEmbedding
  File "F:\iEnviroment\development\python\anaconda3\envs\conda_llm-llamaindex\Lib\site-packages\llama_index\core\base\embeddings\base.py", line 10, in <module>
    import numpy as np
  File "F:\iEnviroment\development\python\anaconda3\envs\conda_llm-llamaindex\Lib\site-packages\numpy\__init__.py", line 125, in <module>
    from numpy.__config__ import show_config
  File "F:\iEnviroment\development\python\anaconda3\envs\conda_llm-llamaindex\Lib\site-packages\numpy\__config__.py", line 4, in <module>
    from numpy._core._multiarray_umath import (
  File "F:\iEnviroment\development\python\anaconda3\envs\conda_llm-llamaindex\Lib\site-packages\numpy\_core\__init__.py", line 24, in <module>
    from . import multiarray
  File "F:\iEnviroment\development\python\anaconda3\envs\conda_llm-llamaindex\Lib\site-packages\numpy\_core\multiarray.py", line 11, in <module>
    from . import _multiarray_umath, overrides
RuntimeError: NumPy was built with baseline optimizations: 
(X86_V2) but your machine doesn't support:
(X86_V2).
```
- 处理
```
pip uninstall numpy -y
pip install numpy --no-binary numpy --config-settings=setup-args="-Dcpu-baseline=none"
```

## agent虚拟环境
```
conda create -n conda_llm-agent python=3.12
conda activate conda_llm-agent
pip install python-dotenv
pip install langchain-tavily
pip install langchain-openai
pip install numexpr

pip install pandas
pip install prettytable

pip install langchain-experimental

pip install pyautogen==0.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## crewai虚拟环境
```
conda create -n conda_crewai python=3.12
conda activate conda_crewai

# 下载对应的包
pip install crewai==1.6.1 crewai-tools==1.6.1 tavily-python==0.7.13 dotenv langchain==0.3.26 langchain-openai==0.3.27 langchain-core==0.3.74 langchain-community==0.3.27 langchain-tavily==0.2.10 dashscope==1.25.2
```

## LangGraph虚拟环境
```
conda create -n conda_llm-langgraph python=3.12
conda activate conda_llm-langgraph

pip install -U "langgraph"

pip install langchain # pip install langchain-core
pip install langchain-community
pip install langchain-deepseek
pip install python-dotenv

pip install ipython

pip install langchain-huggingface
pip install sentence-transformers

pip install -U langgraph-checkpoint-redis

pip install langmem

pip install langchain-tavily
```