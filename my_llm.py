import os
from typing import Optional, List, Dict, Any, Iterator
from openai import OpenAI
from dotenv import load_dotenv

# 从 .env 文件读取 LLM_API_KEY、LLM_MODEL、LLM_BASE_URL
load_dotenv()

class LLM:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")

        if not api_key:
            raise ValueError("需要设置 LLM_API_KEY 环境变量或传入 api_key")

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def ask(self, messages: List[Dict[str, str]], **kwargs) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            **kwargs,
        )
        return resp.choices[0].message.content

    def ask_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
