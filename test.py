import base64
from models import get_claude_opus, get_openai_gpt4, get_together_llama3, get_local_model, get_ollama_local_model, LLM_FN, LLM
from langchain.schema import SystemMessage, AIMessage, HumanMessage
Test_Anthropic = False
Test_OpenAI = True
Test_Together = False
Test_LMStudio = False
Test_Ollama = False
IMAGE_PATH = "images/coke.jpeg" 
# IMAGE_PATH = "images/bison skulls.jpg"
# IMAGE_PATH = "images/sankofa.jpeg"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/e/e2/The_Algebra_of_Mohammed_Ben_Musa_-_page_82b.png"
# Preview image for context
# Open the image file and encode it as a base64 string
ENCODE_FROM_URL = False


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },
]


def test_models():
    llm = None
    if Test_OpenAI:
        llm = LLM(LLM_FN(get_openai_gpt4))
        if not ENCODE_FROM_URL:
            filetype = IMAGE_PATH.split(".")[-1]
            base64_image = encode_image(IMAGE_PATH)
            image_url = f"data:image/{filetype};base64,{base64_image}"
        else:
            image_url = IMAGE_URL
        image_dict = {"url": image_url}
        messages=[
            SystemMessage(content="You are a helpful AI model that analyzes images."),
            HumanMessage(content=[
                {"type": "text", "text": "What is this an image of? Does the image have any red color?"},
                {"type": "image_url", "image_url": image_dict}
            ])
        ]
        response_string = ""
        for chunk in llm.stream(messages):
            response_string += chunk.content
            print(chunk.content, end="")
        print("\nLength of response:", len(response_string))
    if Test_Anthropic:
        llm = LLM(LLM_FN(get_claude_opus))
    if Test_Together:
        llm = LLM(LLM_FN(get_together_llama3))
    if Test_LMStudio:
        llm = LLM(LLM_FN(get_local_model))
    if Test_Ollama:
        llm = LLM(LLM_FN(get_ollama_local_model))
    if llm:
        llm = llm.llm
    print()
    # print("All models are working correctly.")


test_models()
