import yaml
from openai import OpenAI
import anthropic
import requests
import time


# --- Load API keys from YAML ---
with open("config/keys.yaml", "r") as f:
    KEYS = yaml.safe_load(f)

# --- Create clients ---
openai_client = OpenAI(api_key=KEYS.get("openai_key"))
claude_client = anthropic.Anthropic(api_key=KEYS.get("claude_key"))

grok_client = OpenAI(
    api_key=KEYS.get("grok_key"),
    base_url="https://api.x.ai/v1",
    timeout=60.0
)

gemini_client = None  # Gemini uses REST API, not a specific client
deepseek_client = OpenAI(
    api_key=KEYS.get("deepseek_key"),
    base_url="https://api.deepseek.com/v1",
    timeout=60.0
)

mistral_client = OpenAI(
    api_key=KEYS.get("mistral_key"),
    base_url="https://api.mistral.ai/v1",
    timeout=60.0
)


def generate_text(messages, provider="openai", model=None, temperature=0, max_tokens=4096, max_retries=3):
    """
    Generate text using different LLM providers with retry logic.
    
    Args:
        messages: List of {"role": ..., "content": ...}
        provider: "openai", "claude", "gemini", "grok", "deepseek", "mistral"
        model: Model name (provider-specific)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens to generate
        max_retries: Number of times to retry on timeout/rate limit (default: 3)
    
    Returns:
        Generated text string
    
    Raises:
        ValueError: If provider is unknown or API call fails after retries
    """
    
    for attempt in range(max_retries):
        try:
            return _call_llm(messages, provider, model, temperature, max_tokens)
        except (TimeoutError, ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"⚠️  {provider.upper()} timeout/connection error. Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise ValueError(f"Error calling {provider} API after {max_retries} retries: {str(e)}")
        except Exception as e:
            # Don't retry on other errors
            raise ValueError(f"Error calling {provider} API: {str(e)}")


def _call_llm(messages, provider="openai", model=None, temperature=0, max_tokens=4096):
    """Internal function to make LLM API calls."""
    try:
        if provider == "openai":
            model = model or "gpt-4o"
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        elif provider == "claude":
            model = model or "claude-sonnet-4-6"  # Using Claude Sonnet 4
            # Extract system message from messages list (Anthropic API uses 'system' parameter)
            system_msg = None
            api_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    api_messages.append(msg)
            
            # Build API call with system parameter
            api_kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": api_messages,
                "temperature": temperature
            }
            if system_msg:
                api_kwargs["system"] = system_msg
            
            response = claude_client.messages.create(**api_kwargs)
            return response.content[0].text
        
        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=KEYS.get("gemini_key"))
            model_name = model or "gemini-1.5-flash"  # Using Gemini 3 Flash Preview
            
            # Convert messages to Gemini format
            system_prompt = None
            gemini_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    gemini_messages.append({
                        "role": "user" if msg["role"] == "user" else "model",
                        "parts": msg["content"]
                    })
            
            # Create model with system prompt
            model_obj = genai.GenerativeModel(
                model_name,
                system_instruction=system_prompt
            )
            
            response = model_obj.generate_content(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text
        
        elif provider == "grok":
            # xAI Grok - uses current available model
            model = model or "grok-4-1-fast-non-reasoning"
            response = grok_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        elif provider == "deepseek":
            model = model or "deepseek-chat"
            response = deepseek_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        elif provider == "mistral":
            model = model or "mistral-large-latest"
            response = mistral_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: openai, claude, gemini, grok, deepseek, mistral")
    
    except (TimeoutError, ConnectionError, requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        # Re-raise timeout/connection errors for retry logic
        raise
    except Exception as e:
        # Convert other exceptions to ValueError
        raise ValueError(f"Error calling {provider} API: {str(e)}")
