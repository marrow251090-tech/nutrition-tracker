"""Parse ordinary Russian food messages into structured food items."""

import json
from dataclasses import dataclass

from openai import AsyncOpenAI


@dataclass(frozen=True)
class ParsedFoodItem:
    """A food item extracted from a user's message."""

    name: str
    quantity: float
    unit: str
    preparation: str | None = None


@dataclass(frozen=True)
class ParsedFoodMessage:
    """Result of parsing a food message."""

    items: list[ParsedFoodItem]
    needs_clarification: bool = False
    clarification_question: str | None = None


class TextFoodParser:
    """Extract foods and quantities from Russian text using OpenAI."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        reasoning_effort: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort

    async def parse(self, text: str) -> ParsedFoodMessage:
        """Parse a message without calculating nutrition or saving it."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit": {
                                "type": "string",
                                "enum": ["g", "ml", "piece", "serving"],
                            },
                            "preparation": {
                                "type": ["string", "null"],
                            },
                        },
                        "required": [
                            "name",
                            "quantity",
                            "unit",
                            "preparation",
                        ],
                        "additionalProperties": False,
                    },
                },
                "needs_clarification": {"type": "boolean"},
                "clarification_question": {
                    "type": ["string", "null"],
                },
            },
            "required": [
                "items",
                "needs_clarification",
                "clarification_question",
            ],
            "additionalProperties": False,
        }

        instructions = """
Ты разбираешь сообщения пользователя о съеденной еде.
Отвечай только по заданной JSON-схеме.

Правила:
- Понимай русский язык, разговорные фразы и опечатки.
- Выделяй каждый продукт отдельно.
- Сохраняй указанное количество.
- Единицы: g для граммов, ml для миллилитров,
  piece для штук, serving для порций.
- Переводи килограммы в граммы, если это необходимо.
- Для яиц и других продуктов, указанных в штуках,
  используй piece.
- В name указывай конкретный продукт.
- В preparation указывай способ приготовления, если он известен:
  например, варёное, жареное, запечённое, варёная гречка.
- Не придумывай вес, которого пользователь не указал.
- Не считай калории и БЖУ.
- Если количество не указано и его нельзя надёжно определить,
  запроси уточнение.
- Если у гречки, риса, макарон или другого продукта неясно,
  указан вес сухого или готового продукта, запроси уточнение,
  когда это существенно для расчёта.
- Если сообщение не содержит описания еды, верни пустой items
  и запроси уточнение.
- Не считай упоминание продукта доказательством того,
  что пользователь его съел: ориентируйся на смысл сообщения.
"""

        request_payload: dict[str, object] = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": instructions,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "text_food_extract",
                    "strict": True,
                    "schema": schema,
                }
            },
            "store": False,
        }

        if self.reasoning_effort:
            request_payload["reasoning"] = {
                "effort": self.reasoning_effort,
            }

        response = await self.client.responses.create(**request_payload)
        output_text = response.output_text

        if not output_text:
            raise RuntimeError("OpenAI returned an empty response")

        data = json.loads(output_text)

        items = [
            ParsedFoodItem(
                name=str(item["name"]).strip(),
                quantity=float(item["quantity"]),
                unit=str(item["unit"]),
                preparation=item["preparation"],
            )
            for item in data["items"]
            if str(item["name"]).strip() and float(item["quantity"]) > 0
        ]

        return ParsedFoodMessage(
            items=items,
            needs_clarification=bool(data["needs_clarification"]),
            clarification_question=data["clarification_question"],
        )
