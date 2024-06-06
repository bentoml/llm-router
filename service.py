import bentoml
from bentoml.exceptions import BadInput
from openai import AsyncOpenAI

from mistral import MistralService
from toxic_detect import ToxicClassifier

from annotated_types import Ge, Le
from typing_extensions import Annotated
from typing import AsyncGenerator, Literal
from enum import Enum, IntEnum
from pydantic import BaseModel, ValidationError


class ModelName(str, Enum):
    gpt3 = 'gpt-3.5-turbo'
    gpt4 = 'gpt-4o'
    mistral = 'mistral'

MAX_TOKENS = 1024

@bentoml.service(
    traffic={
        "concurrency": 100,
    },
    resources={
        "cpu": "8",
    },
)
class LLMRouter:

    mistral = bentoml.depends(MistralService)
    toxic_classifier = bentoml.depends(ToxicClassifier)

    def __init__(self):
        self.openai_client = AsyncOpenAI()


    @bentoml.api
    async def generate_mistral(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:

        response = self.mistral.generate(prompt, max_tokens)
        async for chunk in response:
            yield chunk

    
    @bentoml.api
    async def generate_openai(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        model: Literal["gpt-3.5-turbo", "gpt-4o"] = "gpt-3.5-turbo",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        res = await self.openai_client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            model=model,
            max_tokens=max_tokens,
            stream=True
        )
        async for chunk in res:
            yield chunk.choices[0].delta.content or ""



    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        model: ModelName = "mistral",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        res: str = self.toxic_classifier.classify([prompt])[0]['label']

        if res == "toxic":
            yield "Bad Input"
        else:
            if model == "mistral":
                gen = self.generate_mistral(prompt, max_tokens)
            else:
                gen = self.generate_openai(prompt, model, max_tokens)

            async for chunk in gen:
                yield chunk
