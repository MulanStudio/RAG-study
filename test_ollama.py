from langchain_ollama import ChatOllama
import os

# Unset proxies just in case
if "http_proxy" in os.environ:
    del os.environ["http_proxy"]
if "https_proxy" in os.environ:
    del os.environ["https_proxy"]

print("Testing Ollama connection...")
try:
    llm = ChatOllama(model="qwen2.5:3b", base_url="http://127.0.0.1:11434")
    response = llm.invoke("Hi")
    print(f"Success! Response: {response.content}")
except Exception as e:
    print(f"Failed: {e}")

