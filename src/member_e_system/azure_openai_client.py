"""
Azure OpenAI client adapters with retry and parallel embedding support
"""

import os
import time
import logging
import concurrent.futures
from typing import List, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator-like function for retry with exponential backoff.
    Handles rate limits (429), timeouts, and transient errors.
    """
    def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if retryable error
                is_rate_limit = "429" in error_str or "rate limit" in error_str
                is_timeout = "timeout" in error_str or "timed out" in error_str
                is_server_error = any(code in error_str for code in ["500", "502", "503", "504"])
                
                if not (is_rate_limit or is_timeout or is_server_error):
                    # Non-retryable error, raise immediately
                    raise e
                
                delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s
                logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
        
        # All retries exhausted
        raise last_exception
    return wrapper


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
        def _call():
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return completion.choices[0].message
        
        return retry_with_backoff(_call)()


class AzureOpenAIEmbeddings:
    """Embedding adapter for LangChain-compatible interface with parallel support"""

    def __init__(self, client: OpenAI, model: str, max_workers: int = 4):
        self.client = client
        self.model = model
        self.max_workers = max_workers
        self.batch_size = 64

    def _clean_texts(self, texts: List[str]) -> List[str]:
        """Clean and validate input texts"""
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
        return cleaned

    def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        """Embed a single batch with retry logic"""
        def _call():
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            return [item.embedding for item in response.data]
        
        return retry_with_backoff(_call)()

    def embed_documents(self, texts: List[str], parallel: bool = True) -> List[List[float]]:
        """
        Embed documents with optional parallel processing.
        
        Args:
            texts: List of texts to embed
            parallel: If True, use parallel batch processing (faster for large datasets)
        """
        cleaned = self._clean_texts(texts)
        
        if len(cleaned) == 0:
            return []
        
        # Split into batches
        batches = [cleaned[i:i + self.batch_size] for i in range(0, len(cleaned), self.batch_size)]
        
        start_all = time.time()
        
        if parallel and len(batches) > 1:
            # Parallel embedding
            return self._embed_parallel(batches, start_all)
        else:
            # Sequential embedding
            return self._embed_sequential(batches, start_all)

    def _embed_sequential(self, batches: List[List[str]], start_all: float) -> List[List[float]]:
        """Sequential batch embedding"""
        embeddings = []
        for i, batch in enumerate(batches):
            start = time.time()
            batch_embeddings = self._embed_batch_with_retry(batch)
            embeddings.extend(batch_embeddings)
            elapsed = time.time() - start
            print(f"   🧩 Embedding batch {i + 1}/{len(batches)}: {len(batch)} docs, {elapsed:.1f}s")
        
        total = time.time() - start_all
        total_docs = sum(len(b) for b in batches)
        print(f"   ✅ Embeddings complete: {total_docs} docs, {total:.1f}s")
        return embeddings

    def _embed_parallel(self, batches: List[List[str]], start_all: float) -> List[List[float]]:
        """Parallel batch embedding using ThreadPoolExecutor"""
        print(f"   🚀 Parallel embedding with {self.max_workers} workers...")
        
        # Results storage (indexed to maintain order)
        results = [None] * len(batches)
        
        def embed_batch_task(idx_batch: Tuple[int, List[str]]) -> Tuple[int, List[List[float]]]:
            idx, batch = idx_batch
            start = time.time()
            embeddings = self._embed_batch_with_retry(batch)
            elapsed = time.time() - start
            print(f"   🧩 Batch {idx + 1}/{len(batches)}: {len(batch)} docs, {elapsed:.1f}s")
            return idx, embeddings
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = list(executor.map(embed_batch_task, enumerate(batches)))
            for idx, embeddings in futures:
                results[idx] = embeddings
        
        # Flatten results
        all_embeddings = []
        for batch_result in results:
            all_embeddings.extend(batch_result)
        
        total = time.time() - start_all
        total_docs = sum(len(b) for b in batches)
        print(f"   ✅ Parallel embeddings complete: {total_docs} docs, {total:.1f}s")
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        t = text if isinstance(text, str) else str(text)
        t = t.strip() or " "
        if len(t) > 4000:
            t = t[:4000]
        
        def _call():
            response = self.client.embeddings.create(
                input=t,
                model=self.model
            )
            return response.data[0].embedding
        
        return retry_with_backoff(_call)()


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
