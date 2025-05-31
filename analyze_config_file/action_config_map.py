
value_list_for_required = [
    "task_config_parameter",
    "task_trigger_parameter",
    "input_file_setting",
    "model_env",
    "model",
    "model_parameter",
    "prompt",
    "prompt_setting",
    "output_setting",
    "other"
]
config_map_for_required = {"anc95/ChatGPT-CodeReview":{
                    "MAX_PATCH_LENGTH": "input_file_setting",
                    "IGNORE_PATTERNS": "input_file_setting",
                    "INCLUDE_PATTERNS": "input_file_setting",
                    "GITHUB_TOKEN": "required_item",
                    "OPENAI_API_KEY": "required_item",
                    "AZURE_API_VERSION": "model_env",
                    "OPENAI_API_ENDPOINT": "model_env",
                    "MODEL": "model",
                    "AZURE_DEPLOYMENT": "model",
                    "top_p": "model_parameter",
                    "temperature": "model_parameter",
                    "max_tokens": "model_parameter",
                    "PROMPT": "prompt",
                    "LANGUAGE": "prompt_setting",
              },
              "mattzcarey/code-review-gpt": {
                    "OPENAI_API_KEY": "required_item",
                    "GITHUB_TOKEN": "required_item",
                    "MODEL": "required_item",
                    "REVIEW_LANGUAGE": "prompt_setting"
              },
              "coderabbitai/ai-pr-reviewer": {
                    "disable_review": "task_config_parameter",
                    "disable_release_notes": "task_config_parameter",
                    "max_files": "input_file_setting",
                    "path_filters": "input_file_setting",
                    "review_simple_changes": "input_file_setting",
                    "GITHUB_TOKEN": "required_item",
                    "OPENAI_API_KEY": "required_item",
                    "openai_base_url": "model_env",
                    "openai_light_model": "model",
                    "openai_heavy_model": "model",
                    "openai_model_temperature": "model_parameter",
                    "system_message": "prompt",
                    "summarize": "prompt",
                    "summarize_release_notes": "prompt",
                    "language": "prompt_setting",
                    "review_comment_lgtm": "output_setting",
                    "debug": "other",
                    "openai_retries": "other",
                    "openai_timeout_ms": "other",
                    "openai_concurrency_limit": "other",
                    "github_concurrency_limit": "other",
                     "bot_icon": "other"
              },
              "aidar-freeed/ai-codereviewer": {
                     "exclude": "input_file_setting",  # Configure exclusion patterns, e.g., **/*.json, **/*.md
                     "GITHUB_TOKEN": "required_item",  # Required GitHub token
                     "OPENAI_API_KEY": "required_item",  # Required OpenAI API key
                     "OPENAI_API_MODEL": "model"  # Optional OpenAI model (default: gpt-4)
               },
              "kxxt/chatgpt-action": {
                     "mode": "task_config_parameter",  # Task mode (pr), only supports pr mode
                     "number": "required_item",  # Required Pull Request or Issue ID
                     "sessionToken": "required_item",  # Required ChatGPT session token
                     "GITHUB_TOKEN": "required_item",  # Required GitHub token
                     "split": "prompt_setting"  # Whether to split prompts and how to split them
               },
              "cirolini/genai-code-review": {
                     "mode": "task_config_parameter",  # Task configuration parameter, determines review mode ('files' or 'patch')
                     "github_pr_id": "required_item",  # Required GitHub PR ID
                     "openai_api_key": "required_item",  # Required OpenAI API key
                     "github_token": "required_item",  # Required GitHub token
                     "openai_model": "model",  # OpenAI language model (gpt-3.5-turbo, gpt-4)
                     "openai_temperature": "model_parameter",  # OpenAI text generation temperature
                     "openai_max_tokens": "model_parameter",  # Maximum token count for OpenAI response
                     "custom_prompt": "prompt",  # Custom prompt
                     "language": "prompt_setting"  # Language for code review (default: "en")
               },
              "truongnh1992/gemini-ai-code-reviewer": {
                     "INPUT_EXCLUDE": "input_file_setting",
                     "OPENAI_TOKEN": "required_item",
                     "GEMINI_API_KEY": "required_item",
                     "GEMINI_MODEL": "model",
              },
              "feiskyer/ChatGPT-Reviewer": {
                     "review_per_file": "task_config_parameter",
                     "GITHUB_TOKEN": "required_item",
                     "OPENAI_API_KEY": "required_item",
                     "OPENAI_API_BASE": "model_env",
                     "model": "model",
                     "temperature": "model_parameter",
                     "frequency_penalty": "model_parameter",
                     "presence_penalty": "model_parameter",
                     "comment_per_file": "output_setting",
                     "blocking": "other"  # Whether to block PR merging
              },
              "adshao/chatgpt-code-review-action": {
                     "FULL_REVIEW_COMMENT": "required_item",  # Review comment content
                     "REVIEW_COMMENT_PREFIX": "required_item",  # Review comment prefix
                     "MAX_CODE_LENGTH": "input_file_setting",  # Code length limit
                     "OPENAI_TOKEN": "required_item",  # OpenAI API token
                     "GITHUB_TOKEN": "required_item",  # GitHub API token
                     "GITHUB_BASE_URL": "model_env",  # GitHub base URL
                     "PROMPT_TEMPLATE": "prompt",  # Prompt template for review
                     "PROGRAMMING_LANGUAGE": "prompt_setting",  # Programming language for code review
                     "ANSWER_TEMPLATE": "output_setting"  # Answer template
              },
              "tmokmss/bedrock-pr-reviewer": {
                     "disable_review": "task_config_parameter",  # Whether to only provide summary without full review
                     "disable_release_notes": "task_config_parameter",  # Whether to disable release notes
                     "max_files": "input_file_setting",  # Maximum number of files to review
                     "path_filters": "input_file_setting",  # Filter specific file paths
                     "review_simple_changes": "input_file_setting",  # Whether to review simple changes
                     "GITHUB_TOKEN": "required_item",  # Required GitHub token
                     "role-to-assume": "required_item",  # Role for AWS OIDC authentication
                     "bedrock_light_model": "model",  # Light model (for simple tasks like summaries)
                     "bedrock_heavy_model": "model",  # Heavy model (for full code review)
                     "bedrock_model_temperature": "model_parameter",  # Text generation temperature
                     "system_message": "prompt",  # System message to control model behavior
                     "summarize": "prompt",  # Final summary prompt
                     "review_file_diff": "prompt",  # Code diff review instruction
                     "summarize_release_notes": "prompt",  # Prompt for generating release notes
                     "language": "prompt_setting",  # Language for code review (ISO language code)
                     "review_comment_lgtm": "output_setting",  # Whether to comment in LGTM cases
                     "debug": "other",  # Whether to enable debug mode
                     "only_allow_collaborator": "other",  # Whether to only allow collaborators to access Bedrock
                     "bedrock_retries": "other",  # Number of retries for Bedrock API timeout or errors
                     "bedrock_timeout_ms": "other",  # Bedrock API request timeout (milliseconds)
                     "bedrock_concurrency_limit": "other",  # Bedrock concurrent API request limit
                     "github_concurrency_limit": "other",  # GitHub concurrent API request limit
                     "bot_icon": "other"  # Bot icon
              },
              "Integral-Healthcare/robin-ai-reviewer": {
                     "files_to_ignore": "input_file_setting",  # Files to ignore (space-separated file list)
                     "GITHUB_TOKEN": "required_item",  # GitHub API token
                     "OPEN_AI_API_KEY": "required_item",  # OpenAI API token
                     "github_api_url": "model_env",  # GitHub server API address (for GitHub Enterprise)
                     "gpt_model_name": "model"  # OpenAI model name (default: gpt-4-turbo)
              },
              "presubmit/ai-reviewer": {
                     "GITHUB_TOKEN": "required_item",  # GitHub API token
                     "LLM_API_KEY": "required_item",  # LLM (Large Language Model) API token
                     "LLM_MODEL": "required_item"  # Large language model to use (e.g., Claude-3-5-Sonnet-20241022)
              },
              "gvasilei/AutoReviewer": {
                     "exclude_files": "input_file_setting",  # Files to exclude (supports wildcard expressions like *.md, *.js)
                     "github_token": "required_item",  # GitHub API token
                     "openai_api_key": "required_item",  # OpenAI API token
                     "model_name": "model",  # OpenAI language model (supports gpt-4 and gpt-3.5-turbo)
                     "model_temperature": "model_parameter"  # Text generation temperature (default: 0)
              },
              "unsafecoerce/chatgpt-action": {
                     "action": "task_config_parameter",  # Action to run, currently supports review and score
                     "path_filters": "input_file_setting",  # File filtering rules for review
                     "GITHUB_TOKEN": "required_item",  # GitHub API token
                     "CHATGPT_ACCESS_TOKEN": "required_item",  # ChatGPT access token (optional, choose one between this and OPENAI_API_KEY)
                     "OPENAI_API_KEY": "required_item",  # OpenAI API token (optional, choose one between this and CHATGPT_ACCESS_TOKEN)
                     "chatgpt_reverse_proxy": "model_env",  # ChatGPT reverse proxy address
                     "review_beginning": "prompt",  # Starting prompt for code review conversation
                     "review_patch": "prompt",  # Review prompt for code blocks/patches
                     "scoring_beginning": "prompt",  # Starting prompt for scoring
                     "scoring": "prompt",  # Prompt for scoring the entire PR
                     "review_comment_lgtm": "output_setting",  # Whether to leave comments even in LGTM (Looks Good to Me) cases
                     "debug": "other"  # Whether to enable debug mode, showing ChatGPT interaction info in CI logs
              },
              "magnificode-ltd/chatgpt-code-reviewer": {
                     "GITHUB_TOKEN": "required_item",  # GitHub API token (for sending review comments)
                     "OPENAI_API_KEY": "required_item",  # OpenAI API token (for calling OpenAI models)
                     "model": "model",  # OpenAI language model to use (default: gpt-3.5-turbo)
                     "max_tokens": "model_parameter"  # Maximum token count for text generation (default: 4096)
              },
              "ca-dp/code-butler": {
                    "comment_body": "task_trigger_parameter",  # Iteration input string (containing issues)
                    "cmd": "required_item",  # Command to execute (supports review/chat)
                    "exclude_files": "input_file_setting",  # Files to exclude (comma-separated)
                    "exclude_extensions": "input_file_setting",  # File extensions to exclude (comma-separated)
                    "GITHUB_TOKEN": "required_item",  # GitHub API token (for accessing GitHub REST API)
                    "OPENAI_API_KEY": "required_item",  # OpenAI API token
                    "model": "model",  # OpenAI language model to use
                    "lang": "prompt_setting"  # Language for generating responses (supports ja or en)
              }
}