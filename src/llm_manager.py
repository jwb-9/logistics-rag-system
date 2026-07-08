"""
LLM管理器
统一管理LLM和嵌入模型
"""
import logging
from typing import Optional, Dict, Any

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from src.config import AppConfig, get_config_manager


logger = logging.getLogger(__name__)


class LLMManager:
    """LLM管理器"""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config_manager().get_config()
        self.llm = None
        self.embeddings = None
        self._initialize_llm()
        self._initialize_embeddings()

    def _initialize_llm(self):
        """初始化LLM"""
        try:
            self.llm = Ollama(
                model=self.config.models.llm,
                base_url=self.config.ollama.base_url,
                temperature=self.config.models.temperature,
                num_predict=self.config.models.max_tokens,
                timeout=self.config.ollama.timeout
            )
            logger.info(f"LLM初始化成功: {self.config.models.llm}")
        except Exception as e:
            logger.error(f"LLM初始化失败: {e}")
            raise

    def _initialize_embeddings(self):
        """初始化嵌入模型"""
        try:
            self.embeddings = OllamaEmbeddings(
                model=self.config.models.embedding,
                base_url=self.config.ollama.base_url,
                # timeout=self.config.ollama.timeout
            )
            logger.info(f"嵌入模型初始化成功: {self.config.models.embedding}")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise

    def get_llm(self) -> Ollama:
        """获取LLM实例"""
        if self.llm is None:
            self._initialize_llm()
        return self.llm

    def get_embeddings(self) -> OllamaEmbeddings:
        """获取嵌入模型"""
        if self.embeddings is None:
            self._initialize_embeddings()
        return self.embeddings

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "llm_model": self.config.models.llm,
            "embedding_model": self.config.models.embedding,
            "temperature": self.config.models.temperature,
            "max_tokens": self.config.models.max_tokens,
            "ollama_url": self.config.ollama.base_url
        }