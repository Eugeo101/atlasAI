# services/agent/llm_client.py
import logging
from typing import Any, Dict, List, Optional
import httpx

from config import AgentConfig

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Base exception for vLLM client communication errors."""
    pass


class LLMClient:
    """Synchronous / Asynchronous transport client for vLLM OpenAI-compatible endpoint."""

    def __init__(self, cfg: Optional[AgentConfig] = None):
        self.cfg = cfg
        self.base_url = self.cfg.vllm_base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.vllm_api_key}",
        }

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Sends a standard chat completion request to vLLM.

        Args:
            messages: List of message dicts (e.g., [{"role": "user", "content": "..."}])
            tools: Optional list of OpenAI-formatted tool schemas.
            tool_choice: "auto", "none", or specific tool requirement.
            temperature: Sampling temperature override.
            max_tokens: Maximum tokens override.

        Returns:
            Dict containing the raw choice payload from vLLM choice object.
        """
        payload: Dict[str, Any] = {
            "model": self.cfg.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.cfg.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.default_max_tokens,
        }

        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        try:
            with httpx.Client(timeout=self.cfg.request_timeout) as client:
                response = client.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                )
                # request = client.build_request(
                #     "POST",
                #     self.endpoint,
                #     headers=self.headers,
                #     json=payload,
                # )

                # print("Request:", request.method, request.url)
                # print("Headers:", {
                #     key: value
                #     for key, value in request.headers.items()
                #     if key.lower() != "authorization"
                # })
                # print("Body:", request.content.decode("utf-8"))

                # response = client.send(request)
                response.raise_for_status()
                data = response.json()

                if "choices" not in data or not data["choices"]:
                    raise LLMClientError(f"Malformed response from vLLM: {data}")

                # Return the choice message object directly (contains role, content, tool_calls)
                return data["choices"][0]["message"]

        except httpx.ConnectError as e:
            logger.error("Failed to connect to vLLM container at %s", self.endpoint)
            raise LLMClientError(f"Cannot reach vLLM service at {self.endpoint}. Is Docker running?") from e
        except httpx.HTTPStatusError as e:
            logger.error("vLLM returned HTTP error %s: %s", e.response.status_code, e.response.text)
            raise LLMClientError(f"vLLM API returned status {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            logger.error("Unexpected error in llm_client: %s", str(e))
            raise LLMClientError(f"Unexpected LLM client error: {str(e)}") from e


if __name__ == "__main__":
    # Quick sanity check execution
    logging.basicConfig(level=logging.INFO)
    config = AgentConfig()
    client = LLMClient(cfg=config)
    
    test_messages = [
        {"role": "system", "content": "You are a concise AI assistant."},
        {"role": "user", "content": "Confirm that the vLLM client connection is working."},
    ]
    
    try:
        logging.info("Testing connection to vLLM at %s...", client.endpoint)
        response_message = client.chat_completion(messages=test_messages)
        print("\n--- vLLM Response ---")
        print(f"Role: {response_message.get('role')}")
        print(f"Content: {response_message.get('content')}")
        print("---------------------\n")
    except LLMClientError as err:
        logging.error("Sanity check failed: %s", err)