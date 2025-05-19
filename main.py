import discord
import re
import asyncio
import random
import groq
from groq import AsyncGroq
from discord import UserFlags
from discord.ui import Button, View
from discord.ext import commands, tasks
from discord import app_commands
from datetime import datetime
from config import *
from logger import logger
from json_stuff import *
from brick_rigs_demo import *
import sys
from tools import *
from core import *
from brick_rigs_demo import *
from PIL import Image
from yt_dlp import YoutubeDL
from datetime import timedelta
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import words
import fitz

# Make sure the word list is downloaded
nltk.download('words')

all_tools = [
    tool_generator("bio", "Adds some text to the user's bio. Only call when requested. The user cannot read their bio, only you can. If something is already in a users bio, do not put it in this as this ADDS stuff to the bio not replaces it.", {
        "tool_input": {"type": "string", "description": "The text to add to the user's bio."}
    }),
    tool_generator("coder", "Executes a python script for tasks such as calculations, data analysis, or problem-solving. The user can't see the script nor the output. GUIs are not supported.", {
        "code": {"type": "string", "description": "The Python code to execute."}
    }),
    tool_generator("web", "Performs a web search or crawls a given URL. Always use this tool for external information or link lookups.", {
        "query": {"type": "string", "description": "The search query or URL to crawl."},
        "sites": {"type": "integer", "description": "Number of top results to return. Usually 3, or 1 for direct URL crawling. Tho if asked to do a deep search or deep dive set this to more than 7."}
    }),
    tool_generator("image", "Generates an image from a prompt. Only call when requested.", {
        "prompt": {"type": "string", "description": "The text prompt to generate the image."},
        "height": {"type": "integer", "description": "Image height."},
        "width": {"type": "integer", "description": "Image width."}
    }),
    tool_generator("plot", "Creates a plot and returns it as an image. Only call when requested.", {
        "data": {"type": "string", "description": "Comma-separated numeric values."},
        "title": {"type": "string", "description": "The title of the plot."},
        "type": {"type": "string", "description": "Plot types: 'line', 'bar', 'pie', 'scatter', 'histogram', 'heatmap' and '3d'."}
    }),
    tool_generator("implode", "Terminates the script of the bot.", {}),
    tool_generator("poll", "Creates a poll.", {
        "title": {"type": "string", "description": "The title of the poll."},
        "options": {"type": "object", "description": "A list of poll options. Eg. [\"Option 1\", \"Option 2\", \"Option 3\"]"},
        "duration": {"type": "integer", "description": "The duration of the poll in seconds."}
    }),
    tool_generator("random", "Generates something random.", {
        "type": {"type": "string", "description": "The type of random to generate. Can be: 'number' or 'word'."}        
    }),
    tool_generator("coin", "Updates the qubicoin balance of a user.", {
        "balances": {"type": "object", "description": "A dictionary of names and their balances. Eg. {\"name1\": 100, \"name2\": 200}"}
    }),
]

# Extract valid tool names from tools list
valid_tool_names = [tool["function"]["name"] for tool in all_tools]

# First, add these classes at the top of the file with your other imports:
class CustomFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

class CustomToolCall:
    def __init__(self, id, type, function):
        self.id = id
        self.type = type
        self.function = CustomFunction(**function)

async def groq_completion(model, messages, tools=None):
    response = ""
    reasoning = ""
    tool_calls = None

    if model == main_model:
        max_tokens = 131072
    else:
        max_tokens = 32768

    completion_args = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "top_p": top_p,
        "max_completion_tokens": max_tokens,
        "stream": True,
    }

    if model == main_model:
        completion_args["reasoning_format"] = "parsed"

    if tools:
        completion_args.update({
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        })

    response_stream = await groq_client.chat.completions.create(**completion_args)
    
    # Print the incremental deltas returned by the LLM.
    async for chunk in response_stream:
        delta = chunk.choices[0].delta
        if delta.reasoning:
            reasoning += delta.reasoning
        if delta.content:
            response += delta.content
        if delta.tool_calls:
            tool_calls = delta.tool_calls

    return response, tool_calls, reasoning

async def generate_response(user_input, name, user_name, user_id, datetime, attachments, processing_msg, required_tools, is_invc, vc_id, bot_invc, links):
    global msg_memory, bio_storage, SYS_MSG, all_tools

    tools = all_tools.copy()

    user_message = f"name: {name}, message: {user_input}"
    
    if is_invc:
        tools += [
            tool_generator("join_vc", "Joins the voice channel the user is currently in. Only call when specifically requested by the user.", {})
        ]
    if bot_invc:
        tools += [
            tool_generator("leave_vc", "Leaves the voice channel the bot is currently in. Only call when specifically requested by the user.", {})
        ]

    sys_msg = ""

    if required_tools:
        sys_msg += f"[Required Tools for this response]: {', '.join(required_tools)}\n"
    if datetime:
        sys_msg += f"\n[Time]: {datetime}"

    sys_msg += SYS_MSG
    
    chat_messages = [
        {"role": "system", "content": sys_msg},
        {"tool_call_id": "users_bio", "role": "tool", "name": "this_users_bio", "content": str(get_bio(bio_storage, user_name))},
        {"tool_call_id": "qubicoins", "role": "tool", "name": "qubicoins", "content": str(qubicoins)},
        *msg_memory,
        {"role": "user", "content": user_message}
    ]

    if attachments:
        # Create a list to store file descriptions
        file_descriptions = []

        for attachment in attachments:
            for extension, content in attachment.items():
                # Decode bytes if necessary
                if isinstance(content, bytes):
                    decoded_content = content.decode('utf-8')  # Adjust encoding if needed
                else:
                    decoded_content = content
                
                # Store the file description
                file_descriptions.append(f"type: {extension}, contents: {decoded_content}")

        # Format the message correctly
        if len(file_descriptions) == 1:
            message_content = f"User sent a {file_descriptions[0]}"
        else:
            message_content = "User sent multiple files: \n" + ". \n".join(
                f"file {i+1}: {desc}" for i, desc in enumerate(file_descriptions)
            )

        # Append the final message
        chat_messages.append({
            "tool_call_id": "file_uploaded",
            "role": "tool",
            "name": "file_uploaded",
            "content": message_content
        })

    if links:
        if len(links) == 1:
            chat_messages.append({
                "tool_call_id": "auto_web",
                "role": "tool",
                "name": "auto_web",
                "content": f"User sent a link, scraping result: {links[0]}"
            })
        else:
            results_text = "User sent links, scraped results:\n"
            results_text += '\n'.join([f"{i+1}: {result}" for i, result in enumerate(links)])
            chat_messages.append({
                "tool_call_id": "auto_web",
                "role": "tool",
                "name": "auto_web",
                "content": results_text
            })

    response = ""
    reasoning = ""
    used_tools = []
    scraped_links = []
    tool_call_texts = []
    tool_call_text = None
    tool_datas = []
    image_tool_result = None
    krill_bot = False
    tool_calls = None
    poll_name = None
    poll_args = None
    poll_duration = None
    model_used = ""

    try:
        response, tool_calls, reasoning = await groq_completion(main_model, chat_messages, tools)
        model_used = main_model
        
        if not response and not tool_calls:
            raise ValueError("No response received from the model")

    except Exception as e:
        logger.error(f"Model '{main_model}' failed with error: {e}")

        if isinstance(e, (groq.RequestTooLarge, groq.BadRequestError)):
            logger.error(f"Model '{main_model}' request too large, error: {e}")
            msg_memory = remove_oldest_responses(msg_memory, 10)
            save_json(msg_memory, MSG_MEMORY_PATH)

            chat_messages = [
                {"role": "system", "content": sys_msg},
                {"tool_call_id": "users_bio", "role": "tool", "name": "this_users_bio", "content": str(get_bio(bio_storage, user_name))},
                *msg_memory,
                {"role": "user", "content": user_message}
            ]

        try:
            # Retry after adjusting chat history
            response, tool_calls, reasoning = await groq_completion(secondary_model, chat_messages, tools)
            model_used = secondary_model

            if not response and not tool_calls:
                response, tool_calls, reasoning = await groq_completion(secondary_model, chat_messages, tools)
                model_used = secondary_model

        except Exception as e:
            logger.error(f"Model '{secondary_model}' failed with error: {e}")
            return None, None, model_used, scraped_links, used_tools, user_message, tool_call_texts, tool_datas, image_tool_result, krill_bot, poll_name, poll_args

    print("reasoning: ", reasoning)
    print("response: ", response)
    print("tool_calls: ",tool_calls)

    if not tool_calls:
        tool_calls = []

    if response:
        function_calls = re.findall(r'<function=(\w+)\[(.*?)\]></function>', response)
        if function_calls:
            for function_name, function_args in function_calls:
                tool_call =  CustomToolCall(
                    id="function_call",
                    type="function",
                    function={
                        "name": function_name,
                        "arguments": function_args
                    }
                )
                tool_calls.append(tool_call)
                response = response.replace(f"<function={function_name}[{function_args}]></function>", "") 

        function_calls = re.findall(r'<function name="(\w+)">({.*?})</function>', response)
        if function_calls:
            for function_name, function_args in function_calls:
                tool_call =  CustomToolCall(
                    id="function_call",
                    type="function",
                    function={
                        "name": function_name,
                        "arguments": function_args
                    }
                )
                tool_calls.append(tool_call)
                response = response.replace(f"<function name=\"{function_name}\">{function_args}</function>", "")
        
        function_calls = re.findall(r'<function=(\w+)\[(.*?)\]>', response)
        if function_calls:
            for function_name, function_args in function_calls:
                tool_call =  CustomToolCall(
                    id="function_call",
                    type="function",
                    function={
                        "name": function_name,
                        "arguments": function_args
                    }
                )
                tool_calls.append(tool_call)
                response = response.replace(f"<function={function_name}[{function_args}]>", "")
        
        function_calls = re.findall(r'<tool_call>(.*?)</tool_call>', response)
        if function_calls:
            for function_call in function_calls:
                function_name = re.search(r'"name":"(\w+)"', function_call)
                function_args = re.search(r'"arguments":({.*?})', function_call)

                if function_name and function_args:
                    tool_call =  CustomToolCall(
                        id="function_call",
                        type="function",
                        function={
                            "name": function_name.group(1),
                            "arguments": function_args.group(1)
                        }
                    )
                    tool_calls.append(tool_call)
                    response = response.replace(f"<tool_call>{function_call}</tool_call>", "")

        function_calls = re.findall(r'<tool_call>(.*?)< /tool_call>', response)
        if function_calls:
            for function_call in function_calls:
                function_name = re.search(r'"name":"(\w+)"', function_call)
                function_args = re.search(r'"arguments":({.*?})', function_call)

                if function_name and function_args:
                    tool_call =  CustomToolCall(
                        id="function_call",
                        type="function",
                        function={
                            "name": function_name.group(1),
                            "arguments": function_args.group(1)
                        }
                    )
                    tool_calls.append(tool_call)
                    response = response.replace(f"<tool_call>{function_call}< /tool_call>", "")
                
    embed = discord.Embed(title="Tool processing...", color=discord.Color.blue(), description="Please wait, Qubicon is working on your request...")
    if tool_calls:
        if reasoning:
            chat_messages.append({
                "role": "assistant",
                "content": reasoning
            })
        else:
            chat_messages.append({
                "role": "assistant",
                "content": response
            })
            
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "bio":
                embed.title = "Adding a string to bio..."
                used_tools.append("bio")
                await processing_msg.edit(embed=embed)

                # check if something is already in the bio
                if function_args["tool_input"] in get_bio(bio_storage, user_name):
                    tool_call_text = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": f"Skipped adding {function_args['tool_input']} to bio for {user_name} as it already is in thier bio.",
                    }

                    tool_data = f"Skipped adding {function_args['tool_input']} to bio for {user_name} as it already is in thier bio."
                    tool_call_texts.append(tool_call_text)
                    tool_datas.append(tool_data)
                    chat_messages.append(tool_call_text)
                    continue

                bio_storage = store_bio(bio_storage, user_name, function_args["tool_input"]+f", timestamp: {datetime}")
                save_json(bio_storage, BIO_PATH)

                tool_call_text = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"Stored {function_args['tool_input']} in bio for {user_name}",
                }

                tool_data = f"Stored {function_args['tool_input']} in bio for {user_name}"

                tool_call_texts.append(tool_call_text)
                tool_datas.append(tool_data)
                
                # Add the tool response to the conversation
                chat_messages.append(tool_call_text)

            elif function_name == "coder":
                embed.title = "Executing some code..."
                used_tools.append("coder")
                await processing_msg.edit(embed=embed)

                python_tool_result = await python_tool(function_args["code"])

                tool_call_text = {
                    "tool_call_id": tool_call.id, 
                    "role": "tool",
                    "name": function_name,
                    "content": f"`{python_tool_result}`, executed python script: ```{function_args['code']}```",
                }

                tool_data = f"Executed python code: \n```\n{function_args['code']}\n```"

                tool_call_texts.append(tool_call_text)
                tool_datas.append(tool_data)
                
                # Add the tool response to the conversation
                chat_messages.append(tool_call_text)

            elif function_name == "web":
                embed.title = f"Searching web for {function_args['query']} and crawling {function_args['sites']} sites..."
                used_tools.append("web")
                await processing_msg.edit(embed=embed)

                search_tool_result = await web_search(function_args["query"], function_args["sites"])

                search_results = json.loads(search_tool_result)
                scraped_link = [result["link"] for result in search_results if "link" in result]
                scraped_links.append(scraped_link)

                tool_call_text = {
                    "tool_call_id": tool_call.id, 
                    "role": "tool",
                    "name": function_name,
                    "content": search_tool_result,
                }

                tool_data = f"Searched web for: \n```\n{function_args['query']}\n```"

                tool_call_texts.append(tool_call_text)
                tool_datas.append(tool_data)
                
                # Add the tool response to the conversation
                chat_messages.append(tool_call_text)
            
            elif function_name == "image":
                embed.title = "Generating an image..."
                used_tools.append("image")
                await processing_msg.edit(embed=embed)

                image_tool_result = await image_tool(function_args["prompt"], function_args["height"], function_args["width"])

                if not image_tool_result == str({"error": "Error processing tool"}):
                    tool_call_text = {
                        "tool_call_id": tool_call.id, 
                        "role": "tool",
                        "name": function_name,
                        "content": f"Image generated succesfully! Prompt: `{function_args["prompt"]}`, size: {function_args['width']}x{function_args['height']}",
                    }
                else:
                    tool_call_text = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": image_tool_result,
                    }

                tool_data = f"Generated image for prompt: \n```\n{function_args['prompt']}\n```"

                tool_call_texts.append(tool_call_text)
                tool_datas.append(tool_data)
                
                # Add the tool response to the conversation
                chat_messages.append(tool_call_text)

            elif function_name == "plot":
                embed.title = "Making a plot..."
                used_tools.append("plot")
                await processing_msg.edit(embed=embed)

                plot_input = function_args["data"]
                values = [float(x.strip()) for x in plot_input.split(',') if x.strip()]
                plot_type = function_args["type"]


                if os.path.exists("downloads/chart.png"):
                    os.remove("downloads/chart.png")
                
                # Reset the plot
                plt.clf()

                if plot_type == 'line':
                    plt.plot(values, marker='o', linestyle='-', color='blue')
                    plt.title(function_args["title"])
                    plt.xlabel("Index")
                    plt.ylabel("Value")
                    plt.grid(True)

                elif plot_type == 'bar':
                    plt.bar(range(len(values)), values, color='green')
                    plt.title(function_args["title"])
                    plt.xlabel("Index")
                    plt.ylabel("Value")
                    plt.grid(True)

                elif plot_type == 'pie':
                    plt.pie(values, labels=[f"Value {i}" for i in range(len(values))], autopct='%1.1f%%')
                    plt.title(function_args["title"])

                elif plot_type == 'scatter':
                    X = np.arange(len(values))
                    Y = values
                    plt.scatter(X, Y, color='red')
                    plt.title(function_args["title"])
                    plt.xlabel("Index")
                    plt.ylabel("Value")
                    plt.grid(True)

                elif plot_type == 'histogram':
                    plt.hist(values, bins=function_args.get("bins", 10), color='purple', edgecolor='black')
                    plt.title(function_args["title"])
                    plt.xlabel("Value")
                    plt.ylabel("Frequency")
                    plt.grid(True)

                elif plot_type == 'heatmap':
                    matrix = np.array(values)
                    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
                    plt.colorbar()
                    plt.title(function_args["title"])
                    plt.xlabel("Column")
                    plt.ylabel("Row")

                elif plot_type == '3d':
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    X = np.arange(len(values))
                    Y = np.random.rand(len(values))  # You can customize Y as needed
                    Z = np.zeros_like(X)
                    dx = dy = 0.5
                    dz = values

                    ax.bar3d(X, Y, Z, dx, dy, dz, color='skyblue')
                    ax.set_title(function_args["title"])
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Value')

                plt.savefig("downloads/chart.png")

                image_tool_result = "downloads/chart.png"

                tool_call_text = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"Ploted data succesfully! Data plotted: `{plot_input}`, type: {plot_type}, name: {function_args['title']}",
                }

                tool_data = f"Generated plot for data: \n```\n{plot_input}\n```, plot type: ```{plot_type}```, plot title: ```{function_args['title']}```"

                tool_call_texts.append(tool_call_text)
                tool_datas.append(tool_data)
                
                # Add the tool response to the conversation
                chat_messages.append(tool_call_text)

            elif function_name == "join_vc":
                embed.title = "Joining voice channel..."
                used_tools.append("join_vc")
                await processing_msg.edit(embed=embed)

                join_vc_tool_result = await join_vc(vc_id)

                tool_call_text = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": join_vc_tool_result,
                }
            
                tool_data = f"Joined voice channel: '{vc_id}'"

                tool_call_texts.append(tool_call_text)
                tool_datas.append(tool_data)

                chat_messages.append(tool_call_text)
            
            elif function_name == "leave_vc":
                embed.title = "Leaving voice channel..."
                used_tools.append("leave_vc")
                await processing_msg.edit(embed=embed)

                leave_vc_tool_result = await leave_vc(vc_id)

                tool_call_text = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": leave_vc_tool_result,
                }
            
                tool_data = f"Left voice channel: '{vc_id}'"

                tool_call_texts.append(tool_call_text)
                tool_datas.append(tool_data)

                chat_messages.append(tool_call_text)

            elif function_name == "shutdown":
                embed.title = "Killing bot..."
                used_tools.append("shutdown")
                await processing_msg.edit(embed=embed)

                krill_bot = True 

                tool_call_text = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"Bot script will be terminated after this response.",
                }
            
                tool_data = f"Bot script will be terminated after this response."

                tool_call_texts.append(tool_call_text)
                tool_datas.append(tool_data)

                chat_messages.append(tool_call_text)
            
            elif function_name == "poll":
                embed.title = "Generating poll..."
                used_tools.append("poll")
                await processing_msg.edit(embed=embed)

                poll_name = function_args["title"]
                poll_args = function_args["options"]
                poll_duration = function_args["duration"]

                tool_call_text = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": "Poll created successfully! It will now be append your response. Poll name: {poll_name}, options: {poll_args}, duration: {poll_duration} seconds",
                }
            
                tool_data = f"Created poll: '{function_args['title']}', options: {function_args['options']}, duration: {function_args['duration']} seconds"

                tool_call_texts.append(tool_call_text)
                tool_datas.append(tool_data)

                chat_messages.append(tool_call_text)

            elif function_name == "random":
                embed.title = "Generating random value..."
                used_tools.append("random")
                await processing_msg.edit(embed=embed)  

                if function_args["type"] == "number":
                    random_tool_result = f"Generated random number: {random.randint(0, 100)}"
                elif function_args["type"] == "word":
                    word_list = words.words()
                    random_tool_result = f"Generated random string: {random.choice(word_list)}"
                else:
                    random_tool_result = f"Invalid random type: {function_args['type']}"

                tool_call_text = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": random_tool_result,
                }

                tool_data = f"Generated random {function_args['type']}."

                tool_call_texts.append(tool_call_text)
                tool_datas.append(tool_data)

                chat_messages.append(tool_call_text)

            elif function_name == "coin":
                embed.title = "Updating qubicoin balance..."
                used_tools.append("coin")
                await processing_msg.edit(embed=embed)

                balances = function_args["balances"]
                qubicoin_tool_result = await update_qubicoin_balances(balances)

                tool_call_text = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": qubicoin_tool_result,
                }

                tool_data = f"Updated qubicoin balance: `{json.dumps(function_args['balances'], indent=2)}`"

                tool_call_texts.append(tool_call_text)
                tool_datas.append(tool_data)

                chat_messages.append(tool_call_text)

        embed.title = "Generating response..."
        await processing_msg.edit(embed=embed)

        response = ""
        try:
            response, tool_calls, reasoning = await groq_completion(main_model, chat_messages, None)
            model_used = main_model
            
            if not response:
                raise ValueError("No response received from the model")

        except Exception as e:
            logger.error(f"Model '{main_model}' failed with error: {e}")

            if isinstance(e, (groq.RequestTooLarge, groq.BadRequestError)):
                logger.error(f"Model '{main_model}' request too large, error: {e}")
                msg_memory = remove_oldest_responses(msg_memory, 10)
                save_json(msg_memory, MSG_MEMORY_PATH)

                chat_messages = [
                    {"role": "system", "content": sys_msg},
                    {"tool_call_id": "users_bio", "role": "tool", "name": "this_users_bio", "content": str(get_bio(bio_storage, user_name))},
                    *msg_memory,
                    {"role": "user", "content": user_message}
                ]

            try:
                # Retry after adjusting chat history
                response, tool_calls, reasoning = await groq_completion(secondary_model, chat_messages, None)
                model_used = secondary_model

                if not response:
                    response, tool_calls, reasoning = await groq_completion(secondary_model, chat_messages, None)
                    model_used = secondary_model

            except Exception as e:
                logger.error(f"Model '{secondary_model}' failed with error: {e}")
                return None, None, model_used, scraped_links, used_tools, user_message, tool_call_texts, tool_datas, image_tool_result, krill_bot, poll_name, poll_args

    if use_reasoning:
        if response:
            # Process <think> and </think> tags in the response
            while "<think>" in response or "</think>" in response:
                # Handle <think>...</think> blocks
                if "<think>" in response:
                    start_idx = response.find("<think>")
                    end_idx = response.find("</think>", start_idx)

                    if end_idx == -1:  # No closing </think>, remove the standalone <think> tag
                        response = response[:start_idx] + response[start_idx + len("<think>"):]
                    else:
                        # Extract content inside the <think> block and remove the entire <think> block
                        cut_content = response[start_idx + len("<think>"):end_idx]
                        response = response[:start_idx] + response[end_idx + len("</think>"):]

                        # Append the extracted content to `reasoning` only if there's actual content
                        if cut_content.strip():
                            reasoning += cut_content.strip()
                # Handle standalone </think> tags
                elif "</think>" in response:
                    end_idx = response.find("</think>")
                    cut_content = response[:end_idx]  # Everything before </think>
                    response = response[end_idx + len("</think>"):]  # Remove everything up to and including </think>

                    # Append the cut content to `reasoning` if there's actual content
                    if cut_content.strip():
                        reasoning += cut_content.strip()

    return response, reasoning, model_used, scraped_links, used_tools, user_message, tool_call_texts, tool_datas, image_tool_result, krill_bot, poll_name, poll_args, poll_duration

@bot.event
async def on_ready():
    logger.info(f'Logged in as {bot.user.name}')
    logger.info(f'ID: {bot.user.id}')
    logger.info('------')

    # Determine time-based greeting
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Good morning"
    elif 12 <= current_hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    # Send startup embed message to designated channel(s)
    for channel in bot.get_all_channels():
        if channel.name == 'qubicon-startup' or channel.id == 1325867568795222128:
            embed = discord.Embed(
                title=f"{greeting}, Qubicon is now online!",
                description=f"Version: {version}, filtering: {do_filter_response}",
                color=discord.Color.green()
            )
            await channel.send(embed=embed)

    await bot.tree.sync()
    check_inactivity.start()

@tasks.loop(seconds=10)
async def check_inactivity():
    current_time = time.time()

    for vc in bot.voice_clients:
        guild_id = vc.guild.id
        last_time = last_audio_time.get(guild_id, 0)
        if current_time - last_time > random.randint(60, 300):
            if not vc.is_playing():
                try:
                    await save_tts_to_wav("z z z z", "vc_tts.wav")
                    audio_source = discord.FFmpegPCMAudio("vc_tts.wav")
                    bot.voice_clients[0].play(audio_source)
                    last_audio_time[guild_id] = time.time()  # Update to avoid spamming
                except Exception as e:
                    print(f"Error in check_inactivity: {e}")

@bot.event
async def on_message(message):
    
    if message.author == bot.user:
        return
    
    if message.author.bot:
        return
    
    if not message.guild:
        return    

    if not message.content and not message.attachments:
        return
    
    if (
        message.channel.name != 'qubicon-general' and
        not (
            message.content.startswith(f'<@{bot.user.id}>') or 
            (
                message.reference and 
                isinstance(message.reference.resolved, discord.Message) and 
                message.reference.resolved.author.id == bot.user.id
            )
        )
    ):
        return

    if message.content.startswith("^"):
        return

    if bot_lockdown:
        if message.author.id not in admins:
            return
        
    if version == "6.0.0":
        await message.reply(content="Feature yet to be added! \n#- Version ")
        return
    
    # Use regex to find tool calls like <toolname> at the beginning
    matches = re.findall(r'^<(\w+)>', message.content)

    tools_to_call = []
    for match in matches:
        if match in valid_tool_names:
            tools_to_call.append(match)

    # Remove tool tags from message content
    message_content = re.sub(r'^<\w+>\s*', '', message.content)

    # regex all user and channel pings
    user_mentions = re.findall(r'<@(\d+)>', message_content)
    channel_mentions = re.findall(r'<#(\d+)>', message_content)
    for user_id in user_mentions:
        user = await bot.fetch_user(user_id)
        if user:
            message_content = message_content.replace(f"<@{user_id}>", f"@{user.display_name}")
    for channel_id in channel_mentions:
        channel = await bot.fetch_channel(channel_id)
        if channel:
            message_content = message_content.replace(f"<#{channel_id}>", f"#{channel.name}")

    try:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        embed = discord.Embed(
            title="Starting processing...",
            description="Please wait, Qubicon is working on your request.",
            color=discord.Color.blue()
        )
        processing_msg = await message.reply(embed=embed)

        attachments = []

        if message.attachments:
            embed.title = "Processing attachments..."
            await processing_msg.edit(embed=embed)

            for attachment in message.attachments:
                # Extract the file extension and content
                extension = attachment.filename.split('.')[-1].lower()
                content = await attachment.read()  # Get the content of the attachment

                if any(attachment.filename.endswith(ext) for ext in image_extensions):
                    decoded_content = await process_image(attachment.url)
                elif any(attachment.filename.endswith(ext) for ext in audio_extensions):
                    decoded_content = await process_audio(message, attachment.url)
                elif extension == "pdf":
                    # Process PDF file
                    with fitz.open(stream=content, filetype="pdf") as doc:
                        decoded_content = "\n".join([page.get_text() for page in doc])
                else:
                    try:
                        decoded_content = content.decode('utf-8')  # default to utf-8
                    except UnicodeDecodeError:
                        decoded_content = "<Unsupported file encoding>"

                # Add each attachment's content to the list in the desired format
                attachments.append({extension: decoded_content})

        links = []

        # regex any http/https links
        http_links = re.findall(r'(https?://[^\s]+)', message_content)
        for link in http_links:
            # scrape the link
            result = AdvCrawler(link).crawl()

            if result:
                links.append(result)
                message_content = message_content.replace(link, "")

        embed.title="Generating response..."
        await processing_msg.edit(embed=embed)

        is_invc = False
        vc_id = None
        bot_invc = False

        if message.author.voice and message.author.voice.channel:
            bot_vc = message.guild.voice_client  # Get bot's voice client for this guild
            if bot_vc and bot_vc.channel:  # Bot is already in a voice channel in this guild
                is_invc = False
                bot_invc = True
                vc_id = bot_vc.channel.id
            else:
                is_invc = True
                vc_id = message.author.voice.channel.id

        response, reasoning, model_used, scraped_links, used_tools, user_message, tool_call_texts, tool_datas, image_tool_result, krill_bot, poll_name, poll_args, poll_duration = await generate_response(
            message_content,
            message.author.display_name,
            message.author.name,
            message.author.id,
            date,
            attachments,
            processing_msg=processing_msg,
            required_tools=tools_to_call,
            is_invc=is_invc,
            vc_id=vc_id,
            bot_invc=bot_invc,
            links=links
        )

        logger.info(f"Reasoning: {reasoning}")
        logger.warning(f"User message: {message_content}")
        logger.info(f"Response: {response}")

        if not response:
            if reasoning:
                response = "Response is empty. Reasoning: " + reasoning
            else:
                embed.title = "Error generating response..."
                embed.description = "An error occurred while generating the response. Please contact an admin."
                embed.color = discord.Color.red()
                await processing_msg.edit(embed=embed)
                return
            
        is_filtered = False    
        if do_filter_response: 
            response, is_filtered = await filter_response(response)

        msg_memory.append({"role": "user", "content": user_message})

        if attachments:
            # Create a list to store file descriptions
            file_descriptions = []

            for attachment in attachments:
                for extension, content in attachment.items():
                    # Decode bytes if necessary
                    if isinstance(content, bytes):
                        decoded_content = content.decode('utf-8')  # Adjust encoding if needed
                    else:
                        decoded_content = content
                    
                    # Store the file description
                    file_descriptions.append(f"type: {extension}, contents: {decoded_content}")

            # Format the message correctly
            if len(file_descriptions) == 1:
                message_content = f"User sent a {file_descriptions[0]}"
            else:
                message_content = "User sent multiple files: \n" + ". \n".join(
                    f"file {i+1}: {desc}" for i, desc in enumerate(file_descriptions)
                )

            # Append the final message
            msg_memory.append({
                "tool_call_id": "call_tpms",
                "role": "tool",
                "name": "file",
                "content": message_content
            })

        if links:
            if len(links) == 1:
                msg_memory.append({
                    "tool_call_id": "auto_web",
                    "role": "tool",
                    "name": "auto_web",
                    "content": f"User sent a link, scraping result: {links[0]}"
                })
            else:
                results_text = "User sent links, scraped results:\n"
                results_text += '\n'.join([f"{i+1}: {result}" for i, result in enumerate(links)])
                msg_memory.append({
                    "tool_call_id": "auto_web",
                    "role": "tool",
                    "name": "auto_web",
                    "content": results_text
                })

        response_to_store = response

        if len(response_to_store) > 4000:
            response_to_store = response_to_store[:3997] + "..."
        
        if reasoning:
            if len(reasoning) > 2000:
                reasoning = reasoning[:1997] + "..."

        msg_memory.append({"role": "assistant", "content": response_to_store})

        # after you’ve changed tool_call_text → tool_call_texts (a list)
        if tool_call_texts:
            for tc in tool_call_texts:
                msg_memory.append(tc)

        save_json(msg_memory, MSG_MEMORY_PATH)

        # regex @userdisplayname and #channelname and turn into <@user_id> and <#channel_id>
        user_mentions = re.findall(r'@(\w+)', response)
        channel_mentions = re.findall(r'#(\w+)', response)
        url_mentions = re.findall(r'\[(.*?)\]\((https?://[^\s]+)\)', response)

        for user_name in user_mentions:
            user = discord.utils.get(message.guild.members, display_name=user_name)
            if user:
                response = response.replace(f"@{user_name}", f"<@{user.id}>")
        for channel_name in channel_mentions:
            channel = discord.utils.get(message.guild.channels, name=channel_name)
            if channel:
                response = response.replace(f"#{channel_name}", f"<#{channel.id}>")
        for url_name, url in url_mentions:
            if not url.startswith("<") and not url.endswith(">"):
                response = response.replace(f"[{url_name}]({url})", f"[{url_name}](<{url}>)")

        # in reasoning too
        if reasoning:
            user_mentions = re.findall(r'@(\w+)', reasoning)
            channel_mentions = re.findall(r'#(\w+)', reasoning)
            url_mentions = re.findall(r'\[(.*?)\]\((https?://[^\s]+)\)', reasoning)
            for user_name in user_mentions:
                user = discord.utils.get(message.guild.members, display_name=user_name)
                if user:
                    reasoning = reasoning.replace(f"@{user_name}", f"<@{user.id}>")
            for channel_name in channel_mentions:
                channel = discord.utils.get(message.guild.channels, name=channel_name)
                if channel:
                    reasoning = reasoning.replace(f"#{channel_name}", f"<#{channel.id}>")
            for url_name, url in url_mentions:
                if not url.startswith("<") and not url.endswith(">"):
                    reasoning = reasoning.replace(f"[{url_name}]({url})", f"[{url_name}](<{url}>)")

        add_response = "\n"
        if scraped_links:
            add_response += "\n-# Scraped links:"
            
            # Flatten list if it's nested
            flat_links = [link for sublist in scraped_links for link in (sublist if isinstance(sublist, list) else [sublist])]
            
            add_response += "".join(f"\n-# - <{link}>" for link in flat_links)
        if used_tools:
            add_response += "\n-# Used tools: " + ", ".join(used_tools)
        if is_filtered:
            add_response += "\n-# Response was filtered."
            
        # Ensure the combined length doesn't exceed 4000 characters
        max_length = 4000
        combined_length = len(response) + len(add_response)
        if combined_length > max_length:
            excess = combined_length - max_length + 3
            response = response[:-excess] + "..."  # Trim the original response

        response += add_response
        
        # Create the button for stats
        stats_button = Button(label="Stats", style=discord.ButtonStyle.grey)

        # Define a callback for the button
        async def stats_callback(interaction):
            if reasoning:
                embed = discord.Embed(
                    title="Reasoning",
                    description=reasoning,
                    color=discord.Color.orange()
                )

                embed.add_field(
                    name="Stats",
                    value=f"Version: {version}\nModel: {model_used}",
                    inline=False
                )
            else:
                embed = discord.Embed(
                    title="Stats",
                    description=f"Version: {version}\nModel: {model_used}",
                    color=discord.Color.orange()
                )

            # if any tools were actually used, list each one
            if used_tools and tool_datas:
                # build a single string with one entry per tool
                tool_input_lines = []
                for i, (tool_name, tool_input) in enumerate(zip(used_tools, tool_datas), start=1):
                    # e.g. "1. bio → Stored ‘foo’ in bio"
                    tool_input_lines.append(f"{i}. **{tool_name}** → {tool_input}")
                tool_inputs_text = "\n".join(tool_input_lines)

                # embed fields have a 1024‑char limit; you can split if necessary
                embed.add_field(
                    name="**Tool input**",
                    value=tool_inputs_text[:1024],  # truncate if over limit
                    inline=False
                )

            await interaction.response.send_message(f"<@{interaction.user.id}>", embed=embed)

        # Set the callback for the button
        stats_button.callback = stats_callback

        # Create a view to hold the button
        view = View()
        view.add_item(stats_button)

        poll = None
        if poll_name:
            # Create a Poll instance
            poll = discord.Poll(
                question=poll_name,
                duration=timedelta(seconds=poll_duration),  # Poll duration set to 1 hour
                multiple=False  # Users can select only one option
            )

            # Add options to the poll
            for option in poll_args:
                poll.add_answer(text=option)

        if len(response) > 2000:
            # Split the response into two parts: first 2000 characters and the rest
            first_part = response[:2000]
            second_part = response[2000:]
            
            # Edit the original message with the first 2000 characters
            await processing_msg.edit(content=first_part, view=view, embed=None)
            
            # Send the rest of the message as a new message
            await message.reply(second_part, poll=poll)
        else:
            # Send the initial response with the button if it's within the character limit
            await processing_msg.edit(content=response, view=view, embed=None)

            if poll:
                await message.reply(poll=poll)

        if image_tool_result:
            await processing_msg.reply(file=discord.File(image_tool_result, "image.jpg"))

        if bot.voice_clients:
            if bot.voice_clients[0].channel.guild.id == message.guild.id:
                # Ensure the bot is connected to the channel
                if bot.voice_clients[0].is_playing() or bot.voice_clients[0].is_paused():
                    print("Already playing or paused")
                else:
                    await save_tts_to_wav(response_to_store, "vc_tts.wav")
                    audio_source = discord.FFmpegPCMAudio("vc_tts.wav")
                    bot.voice_clients[0].play(audio_source)
                    last_audio_time[message.guild.id] = time.time()

        if krill_bot:
            await bot.close()

    except Exception as e:
        logger.error("Failed to generate response, error: " + str(e))
        embed.title = "Error generating response..."
        embed.description = "Houston, we have a problem. \n-# " + str(e)
        embed.color = discord.Color.red()
        await processing_msg.edit(embed=embed)

    await bot.process_commands(message)

# -- Commands --
@bot.tree.command(name="reset_memory", description="Reset the bot's memory (clear chat history).")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.user_install()
async def reset_memory(interaction: discord.Interaction, displ: bool = True):
    """
    Reset the bot's memory (clear message history).

    This command can only be used by the specified admin user.
    """
    
    if interaction.user.id not in admins:
        await interaction.response.send_message(
            "You do not have permission to use this command.", ephemeral=True
        )
        return

    # Clear the chat history in memory
    global msg_memory
    msg_memory = []
    save_json(msg_memory, MSG_MEMORY_PATH)
    
    await interaction.response.send_message("Bot memory has been reset!", ephemeral=False if displ else True)

@bot.tree.command(name="respond", description="Message qubicon directly.")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.user_install()
async def respond(interaction: discord.Interaction, *, message: str):
    embed = discord.Embed(
        title="Starting processing...",
        description="Please wait, Qubicon is working on your request.",
        color=discord.Color.blue()
    )
    
    # You MUST respond to the interaction before using followup
    await interaction.response.defer()

    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Then send a followup message that you can edit later
    processing_msg = await interaction.followup.send(embed=embed, wait=True)

    response, reasoning, model_used, scraped_links, used_tools, user_message, tool_call_texts, tool_datas, image_tool_result = await generate_response(
        message, 
        interaction.user.display_name, 
        interaction.user.name, 
        interaction.user.id, 
        date,
        None, 
        processing_msg, 
        None
    )

    if len(response) > 2000:
        response = response[:1997] + "..."

    await processing_msg.edit(content=response, embed=None)

# Define the autocomplete coroutine
async def demo_autocomplete(interaction: discord.Interaction, current: str):
    options = ["modal", "userinfo"]

    return [
        app_commands.Choice(name=option, value=option)
        for option in options if current.lower() in option.lower()
    ]

@bot.tree.command(name="demo", description="Experimental features, requires knowing their names.")
@app_commands.autocomplete(demo=demo_autocomplete)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def demo(interaction: discord.Interaction, *, demo: str):

    if demo == "modal":
        class DemoModal(discord.ui.Modal, title="Demo Modal"):
            input_field = discord.ui.TextInput(label="Say something", placeholder="Type here...")

            async def on_submit(self, interaction: discord.Interaction):
                await interaction.response.send_message(f"You said: {self.input_field}", ephemeral=True)

        await interaction.response.send_modal(DemoModal())

    elif demo == "userinfo":
        user = interaction.user
        embed = discord.Embed(
            title="User Information",
            description=f"Details for {user.mention}",
            color=discord.Color.blurple()
        )
        embed.set_thumbnail(url=user.display_avatar.url)

        # Banner Image
        if user.banner:
            print("Banner exists")
            embed.set_image(url=user.banner.url)

        # Basic User Info
        embed.add_field(name="Username", value=f"{user.name}#{user.discriminator}", inline=True)
        embed.add_field(name="User ID", value=str(user.id), inline=True)
        embed.add_field(name="Display Name", value=user.display_name, inline=True)
        embed.add_field(name="Bot", value=str(user.bot), inline=True)
        embed.add_field(name="System", value=str(user.system), inline=True)
        embed.add_field(name="Created At", value=user.created_at.strftime('%Y-%m-%d %H:%M:%S'), inline=False)

        # Public Flags (Badges)
        user_flags = user.public_flags
        if user_flags:
            flags = []

            # Dynamically iterate over all available flags
            for flag in UserFlags:
                if user_flags & flag:
                    flags.append(flag.name.replace("_", " ").title())

            # Adding to the embed
            embed.add_field(name="Badges", value=", ".join(flags) if flags else "None", inline=False)

        # Guild-Specific Info (if applicable)
        if interaction.guild:
            member = interaction.guild.get_member(user.id)
            if member:
                # Nickname in the server
                embed.add_field(name="Nickname", value=member.nick if member.nick else "None", inline=True)

                # Server join date
                join_date = member.joined_at.strftime('%Y-%m-%d %H:%M:%S') if member.joined_at else 'Unknown'
                embed.add_field(name="Joined Server", value=join_date, inline=False)

                # Roles in the server (excluding @everyone)
                roles = [role.mention for role in member.roles if role.name != "@everyone"]
                embed.add_field(name="Roles", value=", ".join(roles) if roles else "None", inline=False)

                # Voice State (if in a voice channel)
                if member.voice:
                    voice_channel = member.voice.channel.name if member.voice.channel else "Unknown"
                    mute_status = "Muted" if member.voice.mute else "Unmuted"
                    deafen_status = "Deafened" if member.voice.deaf else "Undeafened"
                    embed.add_field(name="Voice Channel", value=voice_channel, inline=True)
                    embed.add_field(name="Mute Status", value=mute_status, inline=True)
                    embed.add_field(name="Deafen Status", value=deafen_status, inline=True)

                # Activity (if available)
                activity = member.activity
                if activity:
                    activity_type = activity.type.name
                    activity_name = activity.name
                    embed.add_field(name="Current Activity", value=f"{activity_type}: {activity_name}", inline=False)

                # Status (Online, Idle, DND, Offline)
                embed.add_field(name="Status", value=member.status.name.title(), inline=True)

                # Boosting Status
                is_boosting = "Yes" if member.premium_since else "No"
                embed.add_field(name="Server Boosting", value=is_boosting, inline=True)

        # Sending the embed
        await interaction.response.send_message(embed=embed, ephemeral=False)

    else:
        await interaction.response.send_message("You selected an invalid demo.", ephemeral=True)

@bot.tree.command(name="reload", description="Quickly reload some parts of the bot.")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def bot_reload(interaction: discord.Interaction):
    global msg_memory, bio_storage, authorized_users, qubicoins, SYS_MSG, main_model, admins, secondary_model, version, bot_lockdown, use_reasoning, do_filter_response, temp, tts_voice, tts_speed, tts_pitch, tts_volume

    if not interaction.user.id in admins:
        await interaction.followup.send("You do not have permission to use this command.", ephemeral=True)
        return

    # --- memory ---
    msg_memory = load_json(MSG_MEMORY_PATH)
    bio_storage = load_json(BIO_PATH)
    authorized_users = load_json(AGREED_USERS_PATH)
    qubicoins = load_json(QUBICOINS_PATH)
    
    # --- Initial Config Load ---
    cfg = load_config()
    main_model = cfg["main_model"]
    secondary_model = cfg["secondary_model"]
    version = cfg["version"]
    bot_lockdown = cfg["bot_lockdown"]
    use_reasoning = cfg["use_reasoning"]
    do_filter_response = cfg["do_filter_response"]
    temp = cfg["temp"]
    top_p = cfg["top_p"]
    admins = cfg["admins"]
    tts_voice = cfg["tts_voice"]
    tts_speed = cfg["tts_speed"]
    tts_pitch = cfg["tts_pitch"]

    # --- Load System Message ---
    SYS_MSG = ""
    with SYS_MSG_PATH.open('r') as file:
        SYS_MSG = file.read()

    await interaction.response.send_message("Bot reloaded!", ephemeral=True)

@bot.tree.command(name="join", description="Join a voice channel.")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def join(interaction: discord.Interaction):
    channel = interaction.user.voice.channel
    await channel.connect()

    last_audio_time[interaction.guild.id] = time.time()

    await interaction.response.send_message(f"Joined {channel.name}", ephemeral=True)

@bot.tree.command(name="leave", description="Leave a voice channel.")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def leave(interaction: discord.Interaction):
    # Get the bot's voice client for the guild
    voice_client = discord.utils.get(bot.voice_clients, guild=interaction.guild)

    if voice_client and voice_client.is_connected():
        await voice_client.disconnect()
        await interaction.response.send_message("Left the voice channel.", ephemeral=True)
    else:
        await interaction.response.send_message("I'm not connected to a voice channel.", ephemeral=True)

@bot.tree.command(name="lockdown", description="Lock down the bot")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def lockdown(interaction: discord.Interaction):
    global bot_lockdown

    if not interaction.user.id in admins:
        await interaction.followup.send("You do not have permission to use this command.", ephemeral=True)
        return

    if bot_lockdown:
        bot_lockdown = False
        await interaction.response.send_message("Bot is no longer locked down!", ephemeral=False)
        return
    
    bot_lockdown = True
    await interaction.response.send_message("Bot is locked down!", ephemeral=False)

@bot.tree.command(name="filtering", description="Enable/Dissable filtering")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def filtering(interaction: discord.Interaction):
    global do_filter_response

    if not interaction.user.id in admins:
        await interaction.followup.send("You do not have permission to use this command.", ephemeral=True)
        return

    if do_filter_response:
        do_filter_response = False
        await interaction.response.send_message("Bot is not gonna filter responses anymore!", ephemeral=True)
        return
    
    do_filter_response = True
    await interaction.response.send_message("Bot is now gonna filter responses!", ephemeral=True)

@bot.tree.command(name="reasoning", description="Enable/Dissable reasoning")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def reasoning(interaction: discord.Interaction):
    global use_reasoning

    if not interaction.user.id in admins:
        await interaction.followup.send("You do not have permission to use this command.", ephemeral=True)
        return

    if use_reasoning:
        use_reasoning = False
        await interaction.response.send_message("Bot is not gonna reason for responses anymore!", ephemeral=True)
        return
    
    use_reasoning = True
    await interaction.response.send_message("Bot is now gonna reason for responses!", ephemeral=True)

@bot.tree.command(name="read_chnl", description="Read chnl.")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def read_chnl(interaction: discord.Interaction, id: str):
    if not interaction.user.id in admins:
        await interaction.followup.send("Got you, your id has been logged.", ephemeral=True)
        print(f"Logger: {interaction.user.id}")
        return

    channel = bot.get_channel(int(id))
    await interaction.response.send_message(f"Reading {channel.name}", ephemeral=True)

    try:
        async for message in channel.history(limit=100):
            await interaction.followup.send(f"{message.author.display_name}: {message.content}")

            if message.author.id == bot.user.id:
                msg_memory.append({"role": "assistant", "content": message.content})
            else:
                msg_memory.append({"role": "user", "content": f"{message.author.display_name}: {message.content}"})
        
        save_json(msg_memory, MSG_MEMORY_PATH)
    except KeyboardInterrupt:
        save_json(msg_memory, MSG_MEMORY_PATH)
        await interaction.followup.send("Memory saved!", ephemeral=True)
        raise
    except Exception as e:
        await interaction.followup.send(f"Error: {str(e)}", ephemeral=True)
        logger.error(f"Error reading channel {channel.name}: {str(e)}")


@bot.tree.command(name="create_invite", description="Create an non-exipre invite")
@app_commands.allowed_contexts(guilds=True, dms=False, private_channels=True)
async def create_invite(interaction: discord.Interaction):
    invite = await interaction.channel.create_invite(max_age=0, max_uses=0)
    await interaction.response.send_message(f"Heres your invite link! {invite.url}", ephemeral=True)

@bot.tree.command(name="say_vc", description="Make the bot say something in a voice channel.")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def say_vc(interaction: discord.Interaction, *, message: str, mute: bool = False):
    if interaction.user.id not in admins:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)
        return

    await interaction.response.send_message(f"Saying: \"{message}\"", ephemeral=True)

    # Get the voice client for the current guild
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected():
        await interaction.followup.send("I'm not connected to a voice channel.", ephemeral=True)
        return

    await save_tts_to_wav(message, "output.wav")

    if mute:
        try:
            # Mute all members except the bot
            members_to_mute = [m for m in vc.channel.members if m != bot.user]
            await asyncio.gather(*[
                member.edit(mute=True)
                for member in members_to_mute
                if member and member != bot.user
            ], return_exceptions=True)

            vc.play(discord.FFmpegPCMAudio("output.wav"))

            # Wait for playback to finish
            while vc.is_playing():
                await asyncio.sleep(0.5)

        finally:
            await asyncio.gather(*[
                member.edit(mute=False)
                for member in members_to_mute
                if member and member != bot.user
            ], return_exceptions=True)

            # Save the timestamp for cooldowns or rate limiting if needed
            last_audio_time[interaction.guild.id] = time.time()
    else:
        vc.play(discord.FFmpegPCMAudio("output.wav"))
        last_audio_time[interaction.guild.id] = time.time()

@bot.tree.command(name="play", description="Play a song in a voice channel.")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def play(interaction: discord.Interaction, *, link: str):
    if interaction.user.id not in admins:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)
        return

    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.response.send_message("You must be in a voice channel to use this command.", ephemeral=True)
        return

    vc_channel = interaction.user.voice.channel

    # Connect to voice
    if interaction.guild.voice_client is None:
        vc = await vc_channel.connect()
    else:
        vc = interaction.guild.voice_client
        await vc.move_to(vc_channel)

    await interaction.response.send_message("Now playing: " + link, ephemeral=True)

    # ... inside your async play command
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, lambda: ytdl.extract_info(link, download=False))
    url = data['url']
    
    vc.play(discord.FFmpegPCMAudio(url, **ffmpeg_options))
    last_audio_time[interaction.guild.id] = time.time()

@bot.tree.command(name="generate_brv", description="Generate a brv")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def generate_brv(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(ephemeral=True)
    try:
        response, filename = await generate_brci(prompt)

        # Construct the full file path including 'downloads'
        file_path = os.path.join("downloads", filename, "Vehicle.brv")

        # Check if the file exists before sending
        if os.path.exists(file_path):
            await interaction.followup.send(response, file=discord.File(file_path))
        else:
            await interaction.followup.send(response)
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {str(e)}")
        
@bot.tree.command(name="shutup", description="Shut up the bot in vcs")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def shutup(interaction: discord.Interaction):
    if interaction.user.id not in admins:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)
        return

    # Get the voice client for the current guild
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not vc or not vc.is_connected():
        await interaction.response.send_message("I'm not connected to a voice channel.", ephemeral=True)
        return

    # Stop any currently playing audio
    if vc.is_playing():
        vc.stop()

    await interaction.response.send_message("Shutting up!", ephemeral=True)

# command to 

bot.run(QB_TOKEN)