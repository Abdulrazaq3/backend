from openai import OpenAI

client = OpenAI(
    api_key="fw_DVaVggmJPsvK3fKCbB99go",
    base_url="https://api.fireworks.ai/inference/v1"
)

# Test models
models_to_test = [
    "accounts/fireworks/models/qwen2p5-32b-instruct",
    "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "accounts/fireworks/models/qwen3-8b",
]

for model in models_to_test:
    try:
        print(f"Testing: {model}...")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "قل مرحبا"}],
            max_tokens=20,
        )
        print(f"  ✅ OK: {resp.choices[0].message.content}")
        break
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
