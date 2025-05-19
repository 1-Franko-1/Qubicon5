import json
import os
import aiohttp
import io
from pydub import AudioSegment
from io import BytesIO
from logger import *
import re
from config import *
from json_stuff import *
import pydub
import subprocess
import wave
import numpy as np
import time

def remove_oldest_responses(msg_memory, num_to_remove=2):
    # Identify assistant messages with their index (oldest first)
    assistant_indices = [
        i for i, msg in enumerate(msg_memory)
        if msg["role"] == "assistant"
    ]

    # Select the oldest assistant messages to remove
    indices_to_remove = set(assistant_indices[:num_to_remove])

    # Build full removal set
    indices_to_remove_full = set()
    for idx in indices_to_remove:
        indices_to_remove_full.add(idx)  # Remove assistant

        # Remove the following tool message(s), if any
        i = idx + 1
        while i < len(msg_memory) and msg_memory[i]["role"] == "tool":
            indices_to_remove_full.add(i)
            i += 1

        # Optionally, remove preceding user message if it's closely associated
        if idx > 0 and msg_memory[idx - 1]["role"] == "user":
            indices_to_remove_full.add(idx - 1)

    # Also remove the oldest tool message
    for i, msg in enumerate(msg_memory):
        if msg["role"] == "tool":
            indices_to_remove_full.add(i)
            break  # Only the oldest one

    # Return filtered message memory
    return [
        msg for i, msg in enumerate(msg_memory)
        if i not in indices_to_remove_full
    ]

def store_bio(bio: list, user_name: str, bio_text: str):
    for entry in bio:
        if entry[0] == user_name:
            entry[1].append(bio_text)
            return bio
    
    bio.append([user_name, [bio_text]])
    return bio

def get_bio(bio: list, user_name: str):
    for entry in bio:
        if entry[0] == user_name:
            return entry[1]
    return []

def tool_generator(name: str, description: str, parameters: dict):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    param_name: {
                        "type": param_info["type"],
                        "description": param_info["description"]
                    } for param_name, param_info in parameters.items()
                },
                "required": list(parameters.keys()),
                "additionalProperties": False
            },
            "strict": True
        },
    }

async def process_image(url: str, middle_msg: str = ""):
    """
    Process the image from the provided URL, returning a description of the image.

    1. Download the image using aiohttp.
    2. Use a model to process the image and generate a description.
    3. Return the description if the model succeeds, otherwise return an error message.
    """

    # Assuming you're using some model to process the image
    try:
        completion = await groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Describe this image in detail. But keep it short but detailed! {middle_msg}"},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                },
            ],
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=temp,
        )
        return completion.choices[0].message.content if completion and completion.choices else "Error: No description returned from the model."
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return "Error: Failed to process image."

async def download_audio(audio_url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(audio_url) as resp:
            if resp.status != 200:
                return None
            return await resp.read()  # Read audio file as bytes

async def process_audio(message: discord.message, audio_url: str):
    audio_data = await download_audio(audio_url)
    if not audio_data:
        await message.channel.send("Failed to download audio.")
        return None
        
    try:
        # Convert the audio data to an AudioSegment object
        with BytesIO(audio_data) as wav_file:
            audio_segment = AudioSegment.from_wav(wav_file)
        
            # Export the audio to WAV
            wav_io = BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            
            # Process with speech recognition
            with sr.AudioFile(wav_io) as source:
                audio = recognizer.record(source)
                audiotranscription = recognizer.recognize_google(audio)
                return audiotranscription

    except sr.UnknownValueError:
        return "No words detected in the audio."
    except Exception as e:
        return f"Error processing audio: {e}"

async def filter_response(response: str) -> tuple[str, bool]:
    """
    Asynchronously filters the given response string to replace disallowed words 
    with "[ Filtered ]", even if they contain spaces or special characters.
    
    Args:
        response (str): The input string to filter.
    
    Returns:
        tuple[str, bool]: The filtered string and a boolean indicating if filtering occurred.
    """
    # Normalize response by removing non-alphabetic characters (except spaces)
    normalized_response = re.sub(r'[^a-zA-Z\s]', '', response)

    # Create regex pattern that matches words even if they have spaces or special characters between letters
    pattern = re.compile(
        r'\b(' + '|'.join(rf"{r'[^a-zA-Z]*'.join(re.escape(c) for c in word)}" for word in DISALLOWED_WORDS) + r')\b',
        re.IGNORECASE
    )

    # Check if the response contains disallowed words
    isfiltered = bool(pattern.search(normalized_response))

    # Replace matches with "[ Filtered ]"
    filtered_response = pattern.sub("`[ Filtered ]`", response)

    return filtered_response, isfiltered

async def save_tts_to_wav(text: str, filename: str ="output.wav"):
    subprocess.run([
        "espeak-ng",
        "-v", tts_voice,
        "-s", tts_speed,
        "-p", tts_pitch,
        "-w", filename,
        text
    ])

async def join_vc(channel_id):
    try:
        channel = bot.get_channel(channel_id)
        await channel.connect()
        last_audio_time[channel.guild.id] = time.time()
        return f"Joined voice channel '{channel.name}'"
    except Exception as e:
        return f"Failed to join voice channel: {e}"
    
async def leave_vc(channel_id):
    try:
        channel = bot.get_channel(channel_id)
        if not channel:
            return "Channel not found."

        voice_client = discord.utils.get(bot.voice_clients, guild=channel.guild)
        if voice_client and voice_client.is_connected():
            await voice_client.disconnect()
            if channel.guild.id in last_audio_time:
                del last_audio_time[channel.guild.id]
            return f"Left voice channel '{channel.name}'"
        else:
            return "Bot is not connected to a voice channel in this guild."
    except Exception as e:
        return f"Failed to leave voice channel: {e}"
