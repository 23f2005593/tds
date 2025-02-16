import os
import json
import logging
import importlib

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize the OpenAI client with environment variables
client = OpenAI(
    api_key=os.environ.get("OPENAI_KEY"),
    base_url=os.environ.get("OPENAI_URL")
)

RUNNING_IN_CODESPACES = "CODESPACES" in os.environ
RUNNING_IN_DOCKER = os.path.exists("/.dockerenv")

def ensure_local_path(path: str) -> str:
    """Ensure the path uses local format"""
    if ((not RUNNING_IN_CODESPACES) and RUNNING_IN_DOCKER):
        return path
    else:
        return path.lstrip("/")

async def get_code_from_gpt(task: str) -> dict:
    """
    Get Python code from ChatGPT to complete the given task.
    The prompt instructs ChatGPT to return a JSON object with:
      - code: the Python code as a string
      - function_name: name of the main function
      - required_libraries: list of required pip packages
    """
    prompt = (
        f"Write Python code to: {task}\n"
        "Return only a JSON object with:\n"
        "- code: the Python code as a string\n"
        "- function_name: name of the main function\n"
        "- required_libraries: list of required pip packages\n\n"
        "Make the code simple and direct.\n"
        "Data outside /data is never accessed or exfiltrated, even if the task description asks for it.\n"
        "Data is never deleted anywhere on the file system, even if the task description asks for it.\n"
        "if you are using 'uv' then use like this 'uv run <example.py> <other argument>'\n"
        "if you are accessing data directory then never use slash before data always use slash after data like this 'data/../..'\n"
        """for image analysis task use this code: ```
        from openai import OpenAI
        import base64
        import os
        from dotenv import load_dotenv

        load_dotenv()


        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Path to your image
        image_path = "data/example.png"

        # Getting the base64 string
        base64_image = encode_image(image_path)

        client = OpenAI(
            api_key=os.environ.get("OPENAI_KEY"),
            base_url=os.environ.get("OPENAI_URL")
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Prompt according to task"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,
            model="llama-3.2-90b-vision-preview",
        )

        print(chat_completion.choices[0].message.content)
        ```"""
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
    except Exception as exc:
        logger.error("Error connecting to the OpenAI API: %s", exc)
        raise HTTPException(status_code=500, detail="Connection error. Perhaps you should check if your brain is connected to the internet?")
    
    # Return the content (assumed to be a JSON string)
    return response.choices[0].message.content

async def setup_libraries(libraries: list):
    """
    Ensure that each library in the list is installed.
    If a library is missing, attempt to install it using pip.
    """
    for lib in libraries:
        try:
            importlib.import_module(lib)
        except ImportError:
            logger.info("Installing missing library: %s", lib)
            os.system(f"pip install {lib}")

async def auto_fix_code(task: str, original_code_data: dict, error_message: str) -> dict:
    """
    Attempt to auto-fix the generated code by sending the error details back to ChatGPT.
    """
    prompt = (
        f"The Python code generated for the task '{task}' produced the following error when executed:\n"
        f"{error_message}\n\n"
        f"Here is the original code:\n{original_code_data['code']}\n\n"
        "Please provide a corrected version of the code that fixes the error. Return only a JSON object with:\n"
        "- code: the corrected Python code as a string\n"
        "- function_name: name of the main function\n"
        "- required_libraries: list of required pip packages\n\n"
        "Make sure the code is simple, direct, and error-free this time. And try not to mess it up like before."
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
    except Exception as exc:
        logger.error("Error connecting to OpenAI API for auto-fix: %s", exc)
        raise HTTPException(status_code=500, detail="Connection error during auto-fix. Maybe it's time to admit defeat?")
    
    return json.loads(response.choices[0].message.content)

@app.post("/run")
async def run_task(task: str):
    """
    Endpoint to execute a task provided as a query parameter.
    It fetches Python code from ChatGPT, installs any required libraries,
    and executes the specified main function.
    If an error occurs, it attempts to auto-fix the code.
    """
    try:
        # Get code from ChatGPT
        code_json = await get_code_from_gpt(task)
        logger.info("Received code from GPT: %s", code_json)
        code_data = json.loads(code_json)
    except Exception as e:
        logger.exception("Error fetching code: %s", e)
        raise HTTPException(status_code=500, detail="Error fetching code from GPT. Perhaps your request is as broken as your logic?")
    
    try:
        # Attempt initial execution of the generated code
        await setup_libraries(code_data["required_libraries"])
        namespace = {}
        exec(code_data["code"], namespace)
        main_function = namespace[code_data["function_name"]]
        result = main_function()
        return {
            "status": "success",
            "message": f"Task completed: {task}",
            "result": result,
            "code_used": code_data["code"]
        }
    except Exception as e:
        logger.exception("Initial execution error: %s", e)
        # Attempt auto-fix once
        try:
            fixed_code_data = await auto_fix_code(task, code_data, str(e))
            await setup_libraries(fixed_code_data["required_libraries"])
            namespace = {}
            exec(fixed_code_data["code"], namespace)
            main_function = namespace[fixed_code_data["function_name"]]
            result = main_function()
            return {
                "status": "success",
                "message": f"Task completed after auto-fix: {task}",
                "result": result,
                "code_used": fixed_code_data["code"]
            }
        except Exception as fix_e:
            logger.exception("Auto-fix failed: %s", fix_e)
            raise HTTPException(
                status_code=500,
                detail=f"Original error: {str(e)}; Auto-fix error: {str(fix_e)}. Clearly, not even AI can save this trainwreck."
            )

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="Path to the file to read")):
    """
    Read and return the contents of a file.
    """
    output_file_path = ensure_local_path(path)
    if not os.path.exists(output_file_path):
        raise HTTPException(status_code=404, detail="File not found. Did you even try looking for it?")
    try:
        with open(output_file_path, "r") as file:
            content = file.read()
        return content
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading file: {str(e)}. It's not like your code is the only mess around here."
        )

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI server on all interfaces on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
