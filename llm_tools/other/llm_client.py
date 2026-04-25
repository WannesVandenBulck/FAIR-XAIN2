import yaml
from openai import OpenAI
import anthropic
import requests


# --- Load API keys from YAML ---
with open("config/keys.yaml", "r") as f:
    KEYS = yaml.safe_load(f)

# --- Create clients ---
openai_client = OpenAI(api_key=KEYS.get("openai_key"))
claude_client = anthropic.Anthropic(api_key=KEYS.get("claude_key"))

grok_client = OpenAI(
    api_key=KEYS.get("grok_key"),
    base_url="https://api.x.ai/v1"
)

gemini_client = None  # Gemini uses REST API, not a specific client
deepseek_client = OpenAI(
    api_key=KEYS.get("deepseek_key"),
    base_url="https://api.deepseek.com/v1"
)

mistral_client = OpenAI(
    api_key=KEYS.get("mistral_key"),
    base_url="https://api.mistral.ai/v1"
)


def generate_text(messages, provider="openai", model=None, temperature=0.5, max_tokens=4096):
    """
    Generate text using different LLM providers.
    
    Args:
        messages: List of {"role": ..., "content": ...}
        provider: "openai", "claude", "gemini", "grok", "deepseek", "mistral"
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
            model_name = model or "gemini-3-flash-preview"  # Using Gemini 3 Flash Preview
            
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
    
    except Exception as e:
        raise ValueError(f"Error calling {provider} API: {str(e)}")
