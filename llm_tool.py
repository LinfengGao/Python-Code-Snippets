from openai import OpenAI
from anthropic import Anthropic


class LLMTool(object):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        reasoning_effort=None,
        temperature=0,
        top_p=1,
    ):
        """
        :param model_name: str, the name of the model to use, support gpt、claude、gemini、qwen etc.
        :param reasoning_effort: str or bool, the reasoning effort parameter, gpt-5: [None, "minimum", "low", "medium", "high"]; claude: [None, "enabled", "disabled"]; gemini: bool
        :param temperature: float, the temperature parameter, controls the randomness of the output (0 for deterministic output, higher for more randomness). Note: gpt-5 fixed to 1, claude enabled thinking fixed to 1
        :param top_p: float, the top-p sampling parameter, controls the diversity of the output vocabulary
        """
        self.api_key = api_key
        self.base_url = base_url

        self.openai_client = OpenAI(
            api_key=self.api_key,
            base_url=f"{self.base_url}/v1",
            default_headers={"x-foo": "true"},
        )

        self.anthropic_client = Anthropic(
            auth_token=self.api_key,
            base_url=f"{self.base_url}/anthropic/",
            max_retries=0,
        )
        
        if "gpt-5" in model_name:
            assert reasoning_effort in [None, "minimum", "low", "medium", "high"], (
                f"Error reasoning_effort {reasoning_effort} for {model_name}"
            )
        elif "claude" in model_name:
            assert reasoning_effort in [None, "enabled", "disabled"], (
                f"Error reasoning_effort {reasoning_effort} for {model_name}"
            )
        elif "gemini" in model_name:
            assert type(reasoning_effort) == bool, (
                f"Error reasoning_effort {reasoning_effort} for {model_name}"
            )
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.top_p = top_p

    def generate_response(self, content, max_retries=3):
        for i in range(max_retries):
            try:
                if "gpt" in self.model_name:
                    temperature = 1 if "gpt-5" in self.model_name else self.temperature  # gpt-5 only supports temperature=1
                    completion = self.openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": content}],
                        reasoning_effort=self.reasoning_effort,
                        temperature=temperature,
                        top_p=self.top_p,
                    )
                    ret = completion.choices[0].message.content
                elif "claude" in self.model_name:
                    if not self.reasoning_effort or self.reasoning_effort == "disabled":
                        thinking = {"type": "disabled"}
                    else:
                        thinking = {"type": self.reasoning_effort, "budget_tokens": 1024}
                    completion = self.anthropic_client.beta.messages.create(
                        betas=["context-management-2025-06-27"],
                        model=self.model_name,
                        messages=[{"role": "user", "content": content}],
                        thinking=thinking,
                        temperature=1,  # temperature may only be set to 1 when thinking is enabled
                        top_p=self.top_p,
                        max_tokens=2048,
                    )
                    ret = completion.content[-1].text
                elif "gemini" in self.model_name:
                    completion = self.openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": content}],
                        extra_body={"google": {"thinkingConfig": {
                            "includeThoughts": self.reasoning_effort,
                            "thinkingBudget": 512 if self.reasoning_effort else 0
                        }}},
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    ret = completion.choices[0].message.content
                else:
                    completion = self.openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": content}],
                        extra_body={
                            "enable_thinking": str(self.reasoning_effort).lower(), 
                            "preserve_thinking": str(self.reasoning_effort).lower()
                        },
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    ret = completion.choices[0].message.content
                break
            except Exception as e:
                if i == max_retries - 1:
                    raise e
                print(f"Error occurred: {e} in model {self.model_name}")
                print(f"Retrying... {i+1}/{max_retries}")
        return ret, completion
