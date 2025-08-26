from typing import Optional

class BaseProvider:
    def __init__(self, keys: dict):
        self.keys = keys or {}

    def chat(self, model: Optional[str], system: str, user: str, temperature: float, max_tokens: int) -> str:
        raise NotImplementedError

class OpenAIProvider(BaseProvider):
    def chat(self, model: Optional[str], system: str, user: str, temperature: float, max_tokens: int) -> str:
        from openai import OpenAI
        api_key = self.keys.get("openai")
        if not api_key:
            raise RuntimeError("OpenAI key missing")
        client = OpenAI(api_key=api_key)
        model = model or "gpt-4o-mini"
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":user},
            ],
        )
        return resp.choices[0].message.content or ""

class GeminiProvider(BaseProvider):
    def chat(self, model: Optional[str], system: str, user: str, temperature: float, max_tokens: int) -> str:
        import google.generativeai as genai
        api_key = self.keys.get("gemini")
        if not api_key:
            raise RuntimeError("Gemini key missing")
        genai.configure(api_key=api_key)
        model = model or "gemini-1.5-flash"
        prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
        gmodel = genai.GenerativeModel(model)
        resp = gmodel.generate_content(prompt, generation_config={
            "temperature": temperature, "max_output_tokens": max_tokens
        })
        return getattr(resp, "text", "") or ""

class AnthropicProvider(BaseProvider):
    def chat(self, model: Optional[str], system: str, user: str, temperature: float, max_tokens: int) -> str:
        import anthropic
        api_key = self.keys.get("anthropic")
        if not api_key:
            raise RuntimeError("Anthropic key missing")
        client = anthropic.Anthropic(api_key=api_key)
        model = model or "claude-3-haiku-20240307"
        msg = client.messages.create(
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role":"user","content":user}],
        )
        out = []
        for block in msg.content:
            if getattr(block, "type", "") == "text":
                out.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text",""))
        return "\n".join(out)
