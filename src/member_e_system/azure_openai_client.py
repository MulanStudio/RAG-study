"""
Azure OpenAI client adapters
"""

import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv


def create_azure_openai_client(team_domain: str, api_key: str) -> OpenAI:
    base_url = f"https://{team_domain}.cognitiveservices.azure.com/openai/v1/"
    return OpenAI(base_url=base_url, api_key=api_key)


class AzureOpenAIChat:
    """Minimal adapter to provide .invoke() used by generator"""

    def __init__(self, client: OpenAI, model: str, temperature: float = 0.1):
        self.client = client
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt: str):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return completion.choices[0].message


class AzureOpenAIEmbeddings:
    """Embedding adapter for LangChain-compatible interface"""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        cleaned = []
        for t in texts:
            if not isinstance(t, str):
                t = str(t)
            t = t.strip()
            if not t:
                t = " "
            # 避免过长输入导致 400
            if len(t) > 4000:
                t = t[:4000]
            cleaned.append(t)
        embeddings = []
        batch_size = 64
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        t = text if isinstance(text, str) else str(text)
        t = t.strip() or " "
        if len(t) > 4000:
            t = t[:4000]
        response = self.client.embeddings.create(
            input=t,
            model=self.model
        )
        return response.data[0].embedding


def load_azure_settings(config: dict) -> dict:
    load_dotenv()
    azure_cfg = config.get("models", {}).get("azure_openai", {})
    team_domain = azure_cfg.get("team_domain") or os.getenv("TEAM_DOMAIN")
    api_key_env = azure_cfg.get("api_key_env", "AZURE_OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    return {
        "team_domain": team_domain,
        "api_key": api_key,
        "completion_model": azure_cfg.get("completion_model"),
        "completion_model_fallback": azure_cfg.get("completion_model_fallback"),
        "embedding_model": azure_cfg.get("embedding_model"),
    }
