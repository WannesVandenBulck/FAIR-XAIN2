import yaml
from openai import OpenAI
import anthropic


# --- Load API keys from YAML ---
with open("config/keys.yaml", "r") as f:
    KEYS = yaml.safe_load(f)

# --- Create clients ---
openai_client = OpenAI(api_key=KEYS.get("openai_key"))
anthropic_client = anthropic.Anthropic(api_key=KEYS.get("anthropic_key"))

grok_client = OpenAI(
    api_key=KEYS.get("grok_key"),
    base_url="https://api.x.ai/v1"
)


def generate_text(messages, provider="openai", model=None, temperature=0.5, max_tokens=4096):
    """
    Generate text using different LLM providers.
    
    Args:
        messages: List of {"role": ..., "content": ...}
        provider: "openai", "anthropic", "ollama", etc.
        model: Model name (provider-specific)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated text string
    
    Raises:
        ValueError: If provider is unknown or API call fails
    """
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
        
        elif provider == "anthropic":
            model = model or "claude-3-opus-20240229"
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                temperature=temperature
            )
            return response.content[0].text
        
        elif provider == "grok":
            model = model or "grok-3-mini"
            response = grok_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        elif provider == "ollama":
            import requests
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model or "llama2",
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=300
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    except Exception as e:
        raise ValueError(f"Error calling {provider} API: {str(e)}")
