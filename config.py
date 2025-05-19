import os
from pathlib import Path
import discord
from discord.ext import commands
import groq
from groq import AsyncGroq
import speech_recognition as sr
from typing import List, Dict, Any
from json_stuff import *
from tavily import TavilyClient
from yt_dlp import YoutubeDL 

# --- Paths ---
CONFIG_JSON_PATH = Path('data/config.json')
MSG_MEMORY_PATH = Path('data/msg_memory.json')
SYS_MSG_PATH = Path('data/sys_msg.txt')
BIO_PATH = Path('data/bio_storage.json')
AGREED_USERS_PATH = Path('data/agreed.json')
QUBICOINS_PATH = Path('data/qubicoins.json')

# --- Load Config Function ---
def load_config(config_path: Path = CONFIG_JSON_PATH) -> Dict[str, Any]:
    config_data = load_json(config_path)
    return {
        "main_model": config_data.get("main_model", "llama-3.3-70b-versatile"),
        "secondary_model": config_data.get("secondary_model", "llama3-70b-8192"),
        "version": config_data.get("version", "1.0"),
        "bot_lockdown": config_data.get("bot_lockdown", False),
        "use_reasoning": config_data.get("use_reasoning", False),
        "do_filter_response": config_data.get("do_filter_response", True),
        "temp": config_data.get("temp", 0.7),
        "top_p": config_data.get("top_p", 0.7),
        "admins": config_data.get("admins", []),
        "tts_voice": config_data.get("tts_voice", "en-us+m1"),
        "tts_speed": config_data.get("tts_speed", "125"),
        "tts_pitch": config_data.get("tts_pitch", "50"),
    }

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

# --- Environment Variables ---
GROQ_KEY = os.getenv('GROQ')
QB_TOKEN = os.getenv('QB_TOKEN')
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
google_api_key = os.getenv("GOOGLE_KEY")
google_cx = os.getenv("GOOGLE_CX")
tavily_key = os.getenv("TAVILY_KEY")

# --- Constants ---
COMMAND_PREFIX = '!'
image_extensions = {
    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif',
    '.svg', '.heif', '.heic', '.ico', '.jfif', '.raw', '.psd', 
    '.eps', '.ai', '.indd', '.cdr', '.dib', '.emf', '.wmf', '.jp2', 
    '.j2k', '.apng'
}

audio_extensions = [
    ".wav", ".aiff", ".aif", ".pcm", ".au",
    ".flac", ".m4a", ".ape", ".wv", ".tta",
    ".mp3", ".aac", ".ogg", ".opus", ".wma", ".mpc",
    ".mp4", ".mkv", ".avi", ".mov",
    ".m3u", ".pls", ".ts", ".mpd",
    ".mod", ".xm", ".it", ".spc", ".vgm", ".bnsf",
    ".ra", ".rm", ".dvf", ".msv", ".voc", ".amr", ".awb", ".sln", ".gsm", ".dss",
    ".m4r", ".8svx", ".snd", ".caf", ".oga", ".spx", ".loas", ".latm",
    ".sf2", ".sfz", ".brstm", ".hca", ".adx"
]

video_extensions = {
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".mpeg", ".mpg", 
    ".3gp", ".ogv", ".m4v", ".ts", ".vob", ".rm", ".rmvb", ".divx", ".f4v", 
    ".mxf", ".mp2", ".mpv", ".asf", ".mts", ".m2ts", ".dv", ".drc", ".qt"
}

# Define an expanded list of words to filter out in your automoderation system
DISALLOWED_WORDS = [
    "fuck", "shit", "bitch", "asshole", "bastard", "dick", "piss", "cunt",
    "motherfucker", "cock", "prick", "twat", "ballsack", "dildo", "jackass",
    "dipshit", "dumbass", "shithead", "buttfuck", "pussy", "whore", "slut",
    "skank", "nutsack", "tits", "bollocks", "wanker", "bugger", "arse",
    "arsehole", "fucker", "goddamn", "jerkoff", "knobhead", "knob", "shitfuck",
    "shitface", "shitbag", "dickhead", "dickwad", "twatwaffle", "motherfucking",

    # Racial/Ethmic Slurs
    "nigger", "nigga", "chink", "gook", "kike", "spic", "wetback", "coon", "jap", "raghead", 
    "towelhead", "sandnigger", "beaner", "kaffir", "paki", "yid", "mick", 
    "wop", "gypsy", "gyp", "honky", "cracker", "jigaboo", "niglet", "zipperhead",
    "bamboo_coon", "currymuncher", "dothead", "ape", "porchmonkey", "tarbaby",
    "redskin", "squaw", "halfbreed", "mulatto", "oreo", "Uncle Tom"
]

# Youtube_dl options
ytdl_format_options = {
    'format': 'bestaudio/best',
    'noplaylist': True,
    'quiet': True,
    'extractaudio': True,
    'audioformat': 'mp3',
    'outtmpl': 'downloads/%(id)s.%(ext)s',
    'restrictfilenames': True,
    'no_warnings': True,
}

ytdl = YoutubeDL(ytdl_format_options)

ffmpeg_options = {
    'options': '-vn',
}

# --- Discord Bot Setup ---
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.members = True
intents.voice_states = True

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)

# --- Groq API Initialization ---
groq_client = AsyncGroq(api_key=GROQ_KEY, max_retries=2, timeout=10)
web_client = TavilyClient(tavily_key)

recognizer = sr.Recognizer()

# --- memory ---
msg_memory = load_json(MSG_MEMORY_PATH)
bio_storage = load_json(BIO_PATH)
authorized_users = load_json(AGREED_USERS_PATH)
qubicoins = load_json(QUBICOINS_PATH)

last_audio_time = {}

# --- Load System Message ---
SYS_MSG = ""
with SYS_MSG_PATH.open('r') as file:
    SYS_MSG = file.read()
