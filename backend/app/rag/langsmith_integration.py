"""LangSmith 集成配置模块：环境变量配置、Callback Handler 与追踪装饰器。"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from langsmith import Client, traceable
from langchain_core.callbacks import CallbackManager

try:
    # langchain_core < 1.x
    from langchain_core.tracers.langsmith import LangSmithTracer
except ImportError:
    # langchain_core >= 1.x: LangChainTracer 承担同样的 LangSmith 上报职责
    from langchain_core.tracers import LangChainTracer as LangSmithTracer


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
    """LangSmith 管理器：提供追踪器配置、Callback Manager 与追踪装饰器"""

    def __init__(self, config: Optional[LangSmithConfig] = None):
        """config 默认从环境变量加载"""
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
        """创建追踪装饰器（未启用时返回空装饰器）"""
        if not self._enabled:
            def noop_decorator(func):
                return func
            return noop_decorator

        return traceable(
            project_name=self.config.project_name,
            name=name,
            tags=tags or [],
            metadata={"framework": "KnowledgeGraph-RAG"}
        )

    def log_run(self, name: str, inputs: Dict[str, Any],
                outputs: Optional[Dict[str, Any]] = None,
                error: Optional[str] = None,
                tags: Optional[List[str]] = None):
        """手动记录一次运行"""
        if not self._enabled:
            return

        run_data = {
            "name": name,
            "inputs": inputs,
            "outputs": outputs,
            "error": error,
            "tags": tags or []
        }

        if self.client:
            self.client.create_run(**run_data)

    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self._enabled


_langsmith_manager: Optional[LangSmithManager] = None


def get_langsmith_manager() -> LangSmithManager:
    """获取全局 LangSmith 管理器"""
    global _langsmith_manager
    if _langsmith_manager is None:
        _langsmith_manager = LangSmithManager()
    return _langsmith_manager


def traced(name: str, tags: Optional[List[str]] = None):
    """追踪装饰器便捷函数

    用法:
        @traced("query_routing")
        def route_query(query): ...
    """
    manager = get_langsmith_manager()
    return manager.create_tracer(name, tags)


__all__ = [
    'LangSmithConfig',
    'LangSmithManager',
    'LangSmithTracer',
    'get_langsmith_manager',
    'traced',
    'traceable',
]
