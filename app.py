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
        "if you are using 'uv' then use like this 'uv run <example.py> <other argument>'\n"
        "if you are accessing data directory then never use slash before data always use slash after data like this 'data/../..'\n"
        "You are itself language model you dont have to use other llm.\n"
        "You can use bash cammands inside python code to execute task easily.\n"
        "if you can complete task without code then do it.\n"
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
            model="llama-3.2-11b-vision-preview",
        )

        print(chat_completion.choices[0].message.content)
        ```"""
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
    except Exception as exc:
        logger.error("Error connecting to the OpenAI API: %s", exc)
        # Raising an HTTPException will return a JSON error response with status code 500.
        raise HTTPException(status_code=500, detail="Connection error.")

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

@app.post("/run")
async def run_task(task: str):
    """
    Endpoint to execute a task provided as a query parameter.
    It fetches Python code from ChatGPT, installs any required libraries,
    and executes the specified main function.
    """
    try:
        # Get code from ChatGPT
        code_json = await get_code_from_gpt(task)
        logger.info("Received code from GPT: %s", code_json)

        # Parse the JSON response
        code_data = json.loads(code_json)

        # Install required libraries
        await setup_libraries(code_data["required_libraries"])

        # Execute the generated code in a new namespace
        namespace = {}
        exec(code_data["code"], namespace)

        # Run the main function from the generated code
        main_function = namespace[code_data["function_name"]]
        result = main_function()

        return {
            "status": "success",
            "message": f"Task completed: {task}",
            "result": result,
            "code_used": code_data["code"]
        }
    except HTTPException as http_exc:
        # If we already raised an HTTPException, just pass it along.
        raise http_exc
    except Exception as e:
        logger.exception("Error while executing task: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="Path to the file to read")):
    """
    Read and return the contents of a file
    """
    output_file_path = ensure_local_path(path)
    if not os.path.exists(output_file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(output_file_path, "r") as file:
            content = file.read()
        return content
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading file: {str(e)}"
        )        

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI server on all interfaces on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
