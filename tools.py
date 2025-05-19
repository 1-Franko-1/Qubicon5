import aiohttp
import json
import re
import discord
import tempfile
import docker
import os
from crawler import *
import random, time
from config import *
from logger import *
import aiohttp 
import requests

async def python_tool(tool_input: str):
    """Execute code and return the output."""
    try:
        result = run_code_in_docker("python", tool_input)
    
        if "error" in result.lower() or "exception" in result.lower() or "traceback" in result.lower():
            result = f"Your script ran into an issue! Script you provided: ```{tool_input}```, Error: '{result}'"
        else:
            result = f"Your script executed successfully! Output: '{result}'"

        return result
    except Exception as e:
        return json.dumps({"error": "Error processing tool"})

# Define available language images
LANGUAGE_IMAGES = {
    "python": "python:3.10-slim",
}

# Function to detect missing Python dependencies from the code
def detect_python_dependencies(code: str):
    # Regex to find import statements (including submodules like torch.submodule)
    pattern = r"import (\S+)|from (\S+) import"
    imports = re.findall(pattern, code)
    dependencies = set()

    for imp in imports:
        # Only the first part of the tuple will be non-empty, so we take either part
        dependency = imp[0] if imp[0] else imp[1]
        
        # Check for submodule-like imports (torch.submodule, tensorflow.submodule)
        if '.' in dependency:
            base_lib = dependency.split('.')[0]
            dependencies.add(base_lib)
        else:
            dependencies.add(dependency)

    return dependencies

# Function to check if a library is installed in the container environment
def is_library_installed(client, container, library):
    try:
        # Run the command to check if the library is installed
        exec_result = container.exec_run(
            f"python -c 'import {library}'", 
            stderr=True, 
            stdout=True
        )
        return exec_result.exit_code == 0  # Return True if no error, i.e., the library is installed
    except docker.errors.ContainerError as e:
        return False  # Library is not installed if we get a ContainerError

# Function to install dependencies inside a Docker container
def install_python_dependencies(client, container, dependencies):
    for dep in dependencies:
        # Check if the library is already installed before attempting to install
        if not is_library_installed(client, container, dep):
            print(f"Installing {dep}...")
            # Use python -m pip to ensure installation in the right environment
            install_command = ["python", "-m", "pip", "install", dep]
            exec_result = container.exec_run(install_command, stderr=True, stdout=True)
            
            # Check if the installation was successful
            if exec_result.exit_code != 0:
                print(f"Error installing {dep}: {exec_result.output.decode('utf-8')}")
            else:
                print(f"Successfully installed {dep}")
        else:
            print(f"{dep} is already installed, skipping...")

def run_code_in_docker(language, code):
    if language not in LANGUAGE_IMAGES:
        return f"Error: Unsupported language '{language}'. Supported languages are: {list(LANGUAGE_IMAGES.keys())}"

    image = LANGUAGE_IMAGES[language]
    client = docker.from_env()
    container = None
    result = None  # Initialize result to hold the output

    try:
        # Create a temporary directory and script file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Determine the file extension based on the language
            extensions = {
                "python": "py"
            }
            file_extension = extensions.get(language, "txt")  # Default to .txt if unknown
            script_file = os.path.join(temp_dir, f"script.{file_extension}")

            # Write the user code to the script file
            with open(script_file, "w") as f:
                f.write(code)

            # Define the command to execute the script
            command = {
                "python": ["python", f"/app/script.{file_extension}"]
            }.get(language, [])

            if not command:
                return f"Error: Unsupported language '{language}'"

            # Create a container with the selected image
            container = client.containers.run(
                image=image,
                command=["sleep", "infinity"],  # Keep the container running
                detach=True,
                stderr=True,
                stdout=True,
                network_disabled=False,  # Enable network access for package installation
                mem_limit="128m",  # Restrict memory usage
                memswap_limit="256m",  # Total memory + swap limit
                cpu_shares=256,  # Restrict CPU usage
                volumes={temp_dir: {"bind": "/app", "mode": "ro"}},  # Mount script read-only
                working_dir="/app",
                pids_limit=64,  # Limit the number of processes
            )

            # Install dependencies if Python is selected
            if language == "python":
                dependencies = detect_python_dependencies(code)
                install_python_dependencies(client, container, dependencies)

            # Run the user code in the same container
            exec_run = container.exec_run(command, stderr=True, stdout=True)
            result = exec_run.output.decode("utf-8")

    except docker.errors.ContainerError as e:
        result = f"Container error: {e.stderr.decode('utf-8')}"
    except Exception as e:
        result = f"Error: {str(e)}"
    finally:
        # Ensure the container is removed
        if container:
            try:
                container.remove(force=True)
            except Exception as cleanup_error:
                result = f"Error during cleanup: {str(cleanup_error)}"
    
    return result

async def web_search(tool_input: str, num_sites: int) -> str:
    """Perform a web search and return the top results with links. Retries up to 3 times if no results."""
    search_results = []

    try:
        num_sites = int(num_sites)  # Ensure it's an integer

        if 'http://' in tool_input or 'https://' in tool_input:
            # Directly crawl a provided URL
            crawler = AdvCrawler(tool_input)
            scraped_result = crawler.crawl()
            search_results.append({"link": tool_input, "scraped_content": scraped_result})
        else:
            max_retries = 3
            for attempt in range(max_retries):
                print(f"Attempt {attempt + 1}...")
                search_url = f"https://www.googleapis.com/customsearch/v1?q={tool_input}&key={google_api_key}&cx={google_cx}"

                response = requests.get(search_url)
                response.raise_for_status()

                response_json = response.json()
                items = response_json.get("items", [])

                if items:
                    for item in items[:num_sites]:
                        link = item.get("link")
                        crawler = AdvCrawler(link)
                        search_results.append({
                            "title": item.get("title"),
                            "link": link,
                            "snippet": item.get("snippet"),
                            "scraped_content": crawler.crawl()
                        })
                    break
                else:
                    print("No results found, retrying...")

            if not search_results:
                return json.dumps({"error": "No search results after 3 attempts."}, indent=4)

    except requests.RequestException as e:
        return json.dumps({"error": str(e)}, indent=4)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=4)

    return json.dumps(search_results, indent=4)

async def image_tool(image_prompt, height, width):
    try:
        seed = random.randint(1, 100000)

        url = f"https://image.pollinations.ai/prompt/{image_prompt}?width={width}&height={height}&model=flux&seed={seed}"
            
        # Retry mechanism
        attempt = 0
        while attempt < 3:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            # Save the image
                            img_data = await response.read()
                            img_name = os.path.join(
                                'downloads', 
                                f"image-{random.randint(100000000, 999999999)}.jpg"
                            )
                            with open(img_name, 'wb') as file:
                                file.write(img_data)
                            return img_name
                        else:
                            logger.error(f"Failed to download image with status {response.status}")
            except Exception as e:
                logger.error(f"Error downloading image: {e}")

            # Wait before retrying
            attempt += 1
            logger.warning(f"Retrying download for image, attempt {attempt}/{3}...")
            time.sleep(8)  # Wait for 8 seconds before retrying

        # If all retries fail, return None
        return json.dumps({"error": "Error processing tool"})

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return json.dumps({"error": "Error processing tool"})

async def update_qubicoin_balances(balances: dict):
    """
    Updates the QubiCoin balances based on a dictionary.

    Parameters:
    balances (dict): A dictionary like {'username1': '100', 'username2': '2000'}
    """
    if not isinstance(balances, dict):
        return json.dumps({"error": "Invalid data format. Expected a dictionary."})

    qubicoins = load_json(QUBICOINS_PATH)
    if not isinstance(qubicoins, dict):
        qubicoins = {}

    updated_users = []

    for username, balance_str in balances.items():
        try:
            balance = int(balance_str)
            qubicoins[username] = balance
            updated_users.append(f"{username}: {balance}")
        except ValueError:
            return json.dumps({"error": f"Invalid balance for {username}: {balance_str}"})

    save_json(qubicoins, QUBICOINS_PATH)

    return f"Updated balances: {', '.join(updated_users)}"