# Model registry - maps model names to their implementations
MODEL_REGISTRY = {
    # Claude models
    "claude-3-7-sonnet": {
        "client": "anthropic",
        "model_name": "claude-3-7-sonnet-20250219",
        "params": {"max_tokens": 1000, "temperature": 0}
    },
    "claude-3-5-haiku": {
        "client": "anthropic",
        "model_name": "claude-3-5-haiku-20241022",
        "params": {"max_tokens": 1000, "temperature": 0}
    },

    # OpenAI models
    "openai-gpt-4o": {
        "client": "openai",
        "model_name": "gpt-4o", #gpt-4o-2024-08-06
        "params": {"temperature": 0}
    },
    "openai-gpt-4o-mini": {
        "client": "openai",
        "model_name": "gpt-4o-mini", #gpt-4o-mini-2024-07-18
        "params": {"temperature": 0}
    },
    "openai-o3-mini": {
        "client": "openai",
        "model_name": "o3-mini", #o3-mini-2025-01-31
        "params": {"reasoning_effort": "medium"}
    },
    "openai-gpt-4.1": {
        "client": "openai",
        "model_name": "gpt-4.1", #gpt-4_1-2025-04-14
        "params": {"temperature": 0}
    },
    "openai-o4-mini": {
        "client": "openai",
        "model_name": "o4-mini", #o4-mini-2025-04-16
        "params": {"reasoning_effort": "medium"}
    },

    # Azure OpenAI models
    "azure-gpt-4o": {
        "client": "azure",
        "model_name": "gpt-4o",
        "params": {"temperature": 0}
    },
    "azure-gpt-4o-mini": {
        "client": "azure",
        "model_name": "gpt-4o-mini",
        "params": {"temperature": 0}
    },
    "azure-o3-mini": {
        "client": "azure",
        "model_name": "o3-mini",
        "params": {"reasoning_effort": "medium"}
    },

    # Qianfan Llama models
    "llama-3-8b": {
        "client": "qianfan",
        "model_name": "Meta-Llama-3-8B",
        "params": {"temperature": 0.01}
    },
    "llama-3-70b": {
        "client": "qianfan",
        "model_name": "Meta-Llama-3-70B",
        "params": {"temperature": 0.01}
    },
    "llama-3-1-8b": {
        "client": "qianfan",
        "model_name": "Meta-Llama-3.1-8B",
        "params": {"temperature": 0.01}
    },

    # Deepseek models
    "ali-deepseek-v3": {
        "client": "ali",
        "model_name": "deepseek-v3",
        "params": {"temperature": 0}
    },
    "deepseek-v3": {
        "client": "deepseek_native",
        "model_name": "deepseek-chat",
        "params": {"temperature": 0}
    },
    "deepseek-r1": {
        "client": "deepseek_native",
        "model_name": "deepseek-reasoner",
        "params": {"temperature": 0}, # Has no actual effect.
    }
}
