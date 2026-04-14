"""
LangSmith 集成配置模块
用于追踪、调试和评估 RAG 应用

核心功能：
1. 环境变量配置
2. Callback Handler 设置
3. 追踪装饰器
4. 数据集和评估配置
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# LangSmith
from langsmith import Client, traceable
from langchain_core.callbacks import CallbackManager
from langchain_core.tracers.langsmith import LangSmithTracer


@dataclass
class LangSmithConfig:
    """LangSmith 配置"""
    api_key: Optional[str] = None
    project_name: str = "KnowledgeGraph-RAG"
    tracer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> "LangSmithConfig":
        """从环境变量加载配置"""
        return cls(
            api_key=os.environ.get("LANGSMITH_API_KEY"),
            project_name=os.environ.get("LANGSMITH_PROJECT", "knowledge-graph-rag")
        )


class LangSmithManager:
    """
    LangSmith 管理器
    
    提供:
    - 追踪器配置
    - Callback Manager
    - 追踪装饰器
    """
    
    def __init__(self, config: Optional[LangSmithConfig] = None):
        """
        初始化 LangSmith 管理器
        
        Args:
            config: LangSmith 配置，默认从环境变量加载
        """
        self.config = config or LangSmithConfig.from_env()
        self._client: Optional[Client] = None
        self._tracer: Optional[LangSmithTracer] = None
        self._enabled = self._check_enabled()
    
    def _check_enabled(self) -> bool:
        """检查是否启用 LangSmith"""
        if not self.config.api_key:
            print("[LangSmith] 未配置 API Key，追踪功能已禁用")
            return False
        return True
    
    @property
    def client(self) -> Optional[Client]:
        """获取 LangSmith Client"""
        if self._client is None and self._enabled:
            self._client = Client(
                api_key=self.config.api_key,
                auto_tracing=True
            )
        return self._client
    
    @property
    def tracer(self) -> Optional[LangSmithTracer]:
        """获取 LangSmith Tracer"""
        if self._tracer is None and self._enabled:
            self._tracer = LangSmithTracer(
                project_name=self.config.project_name,
                **self.config.tracer_kwargs
            )
        return self._tracer
    
    def get_callback_manager(self) -> Optional[CallbackManager]:
        """获取 Callback Manager"""
        if not self._enabled:
            return None
        return CallbackManager(handlers=[self.tracer])
    
    def create_tracer(self, name: str, tags: Optional[List[str]] = None):
        """
        创建追踪装饰器
        
        Args:
            name: 追踪名称
            tags: 标签列表
        
        Returns:
            traceable 装饰器
        """
        if not self._enabled:
            # 返回空装饰器
            def noop_decorator(func):
                return func
            return noop_decorator
        
        return traceable(
            project_name=self.config.project_name,
            name=name,
            tags=tags or [],
            metadata={"framework": "KnowledgeGraph-RAG"}
        )
    
    def log_run(
        self,
        name: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """
        手动记录一次运行
        
        Args:
            name: 运行名称
            inputs: 输入数据
            outputs: 输出数据
            error: 错误信息
            tags: 标签
        """
        if not self._enabled:
            return
        
        run_data = {
            "name": name,
            "inputs": inputs,
            "outputs": outputs,
            "error": error,
            "tags": tags or []
        }
        
        # 使用 Client 记录
        if self.client:
            self.client.create_run(**run_data)
    
    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self._enabled


# =============================================================================
# 全局 LangSmith 管理器实例
# =============================================================================

_langsmith_manager: Optional[LangSmithManager] = None


def get_langsmith_manager() -> LangSmithManager:
    """获取全局 LangSmith 管理器"""
    global _langsmith_manager
    if _langsmith_manager is None:
        _langsmith_manager = LangSmithManager()
    return _langsmith_manager


# =============================================================================
# 追踪装饰器便捷函数
# =============================================================================

def traced(
    name: str,
    tags: Optional[List[str]] = None
):
    """
    追踪装饰器便捷函数
    
    用法:
        @traced("query_routing")
        def route_query(query):
            ...
    """
    manager = get_langsmith_manager()
    return manager.create_tracer(name, tags)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'LangSmithConfig',
    'LangSmithManager',
    'LangSmithTracer',
    'get_langsmith_manager',
    'traced',
    'traceable',
]
