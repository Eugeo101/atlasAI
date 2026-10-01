import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Replace with your vLLM URL (e.g. "http://localhost:8000/v1" or Colab tunnel URL)
load_dotenv('.env.example')
VLLM_BASE_URL = "https://characters-outreach-deals-holdem.trycloudflare.com/v1"
GOLDEN_SET_PATH = "golden-set-v1.json"  # Path to your dataset

# ---------------------------------------------------------------------------
# Helper: Load First Arabic Sample
# ---------------------------------------------------------------------------
def load_first_sample(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: '{file_path}' not found. Using default Arabic test query.")
        return {
            "prompt": "ما هي الخطوات الأساسية للبدء في تعلم الذكاء الاصطناعي؟",
            "response": "الخطوات الأساسية تتضمن تعلم البرمجة بلغة ب think, تعلم الرياضيات والخوارزميات..."
        }
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Handle list of dicts or dict with 'goldens' key
        items = data if isinstance(data, list) else data.get("goldens", data.get("data", []))
        if not items:
            raise ValueError("Golden set file is empty!")
        return items[0]

# ---------------------------------------------------------------------------
# Main Smoke Test
# ---------------------------------------------------------------------------
def run_smoke_test():
    sample = load_first_sample(GOLDEN_SET_PATH)
    
    user_query = sample.get("prompt")
    expected = sample.get("response")
    
    print("=" * 60)
    print("🧪 RUNNING ARABIC SMOKE TEST ON CURRENT VLLM MODEL")
    print(f"🌐 vLLM Endpoint: {VLLM_BASE_URL}")
    print("=" * 60)
    print(f"\n📥 USER QUERY (Arabic):\n{user_query}\n")
    if expected:
        print(f"🎯 EXPECTED OUTPUT:\n{expected}\n")
    
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
    
    # Query vLLM active model
    try:
        # Fetch dynamic model name from vLLM endpoint
        models = client.models.list()
        model_name = models.data[0].id if models.data else "default"
        print(f"🤖 Loaded Model in vLLM: {model_name}")
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "أنت مساعد ذكي ومفيد. أجب باللغة العربية بوضوح ودقة."},
                {"role": "user", "content": user_query}
            ],
            temperature=0.2,
            max_tokens=512
        )
        latency = time.time() - start_time
        output_text = response.choices[0].message.content

        print("-" * 60)
        print(f"📤 MODEL OUTPUT (Generated in {latency:.2f}s):\n")
        print(output_text)
        
        print("-" * 60)
        
        # Manual Check Checklist
        print("\n🔍 QUICK VERIFICATION CHECKLIST:")
        print(" [ ] Did the model output valid Arabic text (not gibberish / English)?")
        print(" [ ] Did it follow the instruction / system prompt?")
        print(" [ ] Is the response length and structure coherent?")
        
    except OpenAIError as e:
        print(f"❌ Connection error or vLLM failure: {e}")

if __name__ == "__main__":
    run_smoke_test()