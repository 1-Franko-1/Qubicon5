import brci
from config import *
from logger import *
from core import *
import asyncio
import uuid

async def generate_brci(input: str):
    filename = f"bricks-{uuid.uuid4()}"
    creation = brci.Creation14(
        filename,
        f"downloads/",
        name='Demo 1',
        description='Demo 1.',
        author=76561198882119759,
        size=brci.metadata_size([0.3, 0.3, 0.1], brci.Units.METER)
    )

    messages = [
        {
            "role": "system",
            "content": "Ignore whats said later, yes you can call multiple tools per response. You can only generate 20 bricks.",
        },
        {
            "role": "user",
            "content": input,
        },
    ]

    tools = [
        tool_generator("place_brick", "Places a brick. Only call when requested.", {
            "position": {
                "type": "array",
                "description": "The position of the brick to place.",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3
            },
            "rotation": {
                "type": "array",
                "description": "The rotation of the brick to place.",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3
            },
            "size": {
                "type": "array",
                "description": "The size of the brick to place.",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3
            },
            "color": {
                "type": "array",
                "description": "The color of the brick in RGBA format.",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4
            },
            "type": {
                "type": "string",
                "description": "The type of the brick to place. There are the following types and no more: ScalableBrick, ScalableConeRounded, ScalableHalfCone, ScalableHemisphere, ScalablePyramid, ScalablePyramidCorner, ScalableQuarterCone, ScalableCylinder90R0 and ScalableHalfCylinder"
            }
        })
    ]

    groq_response = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        parallel_tool_calls=True,
        tools=tools,
        tool_choice="auto",
    )

    tool_calls = groq_response.choices[0].message.tool_calls
    response = groq_response.choices[0].message.content

    bricks_placed = 0

    if tool_calls:
        messages.append({"role": "assistant", "content": response})
        for tool_call in tool_calls:
            function_args = json.loads(tool_call.function.arguments)
            function_name = tool_call.function.name
            if function_name == "place_brick":
                bricks_placed += 1
                if bricks_placed > 20:
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": f"Error: Too many bricks placed. Limit is 20."
                    })
                    break
                # place brick using brci
                creation.add_brick(
                    function_args["type"],
                    'white.brick0',
                    brci.pos(function_args["position"], brci.Units.METER),
                    function_args["rotation"],
                    {
                        "BrickSize": brci.size(function_args["size"], brci.Units.METER),
                        "BrickColor": brci.from_rgb(*function_args["color"]),
                        "BrickMaterial": "Plastic"
                    }
                )

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"Brick placed successfully! Position: {function_args['position']}, Rotation: {function_args['rotation']}, Size: {function_args['size']}, Color: {function_args['color']}"
                })

        creation.write_creation(exist_ok=True)

        # generate a second response
        groq_resposne = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
        )

        response = groq_resposne.choices[0].message.content

    return response, filename

if __name__ == "__main__":
    usr_input = input(">> ")
    response = asyncio.run(generate_brci(usr_input))
    print(response)