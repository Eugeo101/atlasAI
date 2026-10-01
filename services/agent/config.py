# services/agent/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

# Go 2 levels up from current file, into target folder, to target file
target_dir = Path(__file__).resolve().parents[2]
load_dotenv(target_dir / '.env.example')

@dataclass
class AgentConfig:
    vllm_base_url: str = os.getenv("VLLM_BASE_URL")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "EMPTY")
    model_name: str = os.getenv("MODEL_NAME")
    request_timeout: float = float(os.getenv("GATEWAY_TIMEOUT"))
    max_tool_rounds: int = 3
    default_temperature: float = 0.1
    default_max_tokens: int = 1024

if __name__ == '__main__':
    config = AgentConfig()
    print(config)