# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from llama_cpp import Llama


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.llm = Llama("./umbra-moe-4x10.7.gguf")

    def predict(
        self,
        prompt: str = Input(description="The prompt to generate text from.",default="Tell me a random fact about the universe."),
        max_tokens: int = Input(
          description="The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.",
      		default=128
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        stream = self.llm(prompt=prompt, max_tokens=128, stop=["\n"], stream=True)
        for output in stream:
            yield output['choices'][0]['text']
