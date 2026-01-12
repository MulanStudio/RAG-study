import yaml
import streamlit as st

def load_config():
    """加载配置文件"""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"无法加载 config.yaml: {e}")
        return None

# --- 然后修改 streamlit_app.py 中的硬编码部分 ---
# 例如：
# chunk_size=2000 -> chunk_size=config["indexing"]["chunk_size_parent"]
# prompt = "..." -> prompt = config["prompts"]["generation"]

