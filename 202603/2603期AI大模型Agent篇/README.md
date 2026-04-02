# 虚拟环境
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