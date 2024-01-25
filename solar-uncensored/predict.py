# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, ConcatenateIterator
from llama_cpp import Llama


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.llm = Llama("./solar-10.7b-instruct-v1.0-uncensored.Q5_K_M.gguf")

    def predict(
        self,
        prompt: str = Input(
            description="The prompt to generate text from.",
            default="Tell me a random fact about the universe. Did you know that ",
        ),
        max_tokens: int = Input(
            description="The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.",
            default=128,
        ),
        temperature: float = Input(
            description="The temperature to use for sampling.", default=0.8
        ),
        top_p: float = Input(
            description="The top-p value to use for nucleus sampling.", default=0.95
        ),
        top_k: int = Input(
            description="The top-k value to use for top-k sampling.", default=40
        ),
        presence_penalty: float = Input(
            description="The penalty to apply to tokens based on their presence in the prompt.",
            default=0.0,
        ),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        stream = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            stop=["\n"],
            stream=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
        )
        for output in stream:
            yield output["choices"][0]["text"]
