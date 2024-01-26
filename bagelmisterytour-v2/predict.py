# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, ConcatenateIterator
from llama_cpp import Llama


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.llm = Llama(
            "./BagelMIsteryTour-v2-8x7B.Q4_K_M.gguf",
            n_gpu_layers=-1,
            n_ctx=2048,
        )

    def predict(
        self,
        prompt: str = Input(
            description="The prompt to generate text from.",
            default="""<s> ### User:
What's the largest planet in the solar system?

### Assistant:
""",
        ),
        max_tokens: int = Input(
            description="The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.",
            default=16,
        ),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        stream = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            stop=["\n"],
            stream=True,
        )
        for output in stream:
            yield output["choices"][0]["text"]
