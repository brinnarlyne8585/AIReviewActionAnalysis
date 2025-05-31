
# Default timeouts and retry settings
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 30


# Client configuration for different LLM providers
CLIENT_CONFIGS = {
    "anthropic": {
        "type": "anthropic",
        "params": {
            "api_key": '',
        }
    },
    "openai": {
        "type": "openai",
        "params": {
            "api_key": "",
            "base_url": "https://api.openai.com/v1"
        }
    },
    "azure": {
        "type": "azure_openai",
        "params": {
            "azure_endpoint": "https://subcription-openai.openai.azure.com/",
            "api_key": "",
            "api_version": "2024-09-01-preview"
        }
    },
    "qianfan": {
        "type": "qianfan",
        "params": {
            "ak": "",
            "sk": ""
        }
    },
    "ali": {
        "type": "openai_compatible",
        "params": {
            "api_key": "",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
        }
    },
    "deepseek_native": {
        "type": "openai_compatible",
        "params": {
            "api_key": "",
            "base_url": "https://api.deepseek.com"
        }
    }
}

