import base64
# from models.models import get_claude_opus, get_ollama_mistral, get_openai_gpt4, get_together_llama3, get_local_model, get_ollama_llama3, LLM_FN, LLM
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
"""
Output a valid json object for each excerpt (4 in total). It should contain the source, index, and a boolean value for criteria_met. If the criteria is met, provide a snippet of what it mentions, and what criteria it corresponds to. CRITERIA: Any word or phrase related to the 1) brain, 2) cognitive science, or 3) research.

Example 1:
context: "The study, published in the journal Science, used functional magnetic resonance imaging (fMRI) to monitor the brain activity of participants while they performed cognitive tasks."

output: {
  "source": "Research_Paper_12.pdf",
  "index": 1,
  "criteria_met": True,
  "mention": "brain activity",
  "criteria": "brain",
}

Example 2:
context: "Cognitive science is a multidisciplinary field of study that draws from neuroscience, psychology, philosophy, computer science, anthropology, and linguistics."

output: {
  "source": "Cognitive_Science_Overview.txt",
  "index": 2,
  "criteria_met": True,
  "mention": "Cognitive science",
  "criteria": "cognitive science",
}

Example 3:
context: "The research team conducted a series of experiments to investigate the neural mechanisms underlying decision-making processes."

output: {
  "source": "Neural_Mechanisms_Study.docx",
  "index": 3,
  "criteria_met": True,
  "mention": "research team",
  "criteria": "research",
}

Example 4:
context: "The book explores the fascinating world of artificial intelligence and its potential impact on society."

output: {
  "source": "AI_Book.epub",
  "index": 4,
  "criteria_met": False,
  "mention": "",
  "criteria": "",
}

Example 5:
context: "The lecture covered various aspects of human memory, including short-term memory, long-term memory, and working memory."

output: {
  "source": "Memory_Lecture.pptx",
  "index": 5,
  "criteria_met": True,
  "mention": "human memory",
  "criteria": "brain",
}
Now provide an output object for the given context.
"""
# This file is in Augmenta/framework/tests
# The root is Augmenta/framework
# How can I import the Chatbot class from chat.py and Config class from config.py?
from chat import Chatbot
from classes import Config
import unittest
from utils import read_sample
from chains import SimpleChain, get_eval_chain
from models.models import LLM, LLM_FN
from models.models import get_local_model, get_together_bigmix, get_together_qwen, get_together_llama3, get_together_dbrx, get_openai_gpt4
from models.models import get_openai_gpt4, get_together_dolphin, get_together_nous_mix, get_together_arctic, get_together_fn_mix, get_claude_sonnet

MODEL_FN = get_claude_sonnet
# class TestChatbot(unittest.TestCase):
#     def setUp(self):
#         config_override = {
#             "chat": {
#                 "primary_model": "local",
#             },
#             "RAG": {
#                 "rag_mode": False,
#                 "collection_name": "0",
#             }
#         }
#         self.chatbot = Chatbot(Config(config_override))

#     def test_get_chat_response(self):
#         prompt = "Capital of indiana?"
#         response = self.chatbot.get_chat_response(prompt, stream=False)
#         self.assertIsInstance(response, AIMessage)

    # def test_chatbot_nonpersistent(self):
    #     messages = self.chatbot.chat(".read", persistence_enabled=False)
    #     self.assertIsInstance(messages, list)
    #     response = messages[-1]
    #     self.assertIsInstance(response, AIMessage)

    # def test_get_rag_response(self):
    #     self.chatbot.config.rag_settings.rag_mode = True
    #     self.chatbot.initialize_rag()
    #     response = self.chatbot.get_rag_response("What is RAG?", stream=False)
    #     self.assertIsInstance(response, AIMessage)
# class EvalTest:
#     def __init__(self, chain: SimpleChain, required_keys = [], validity_fn = None, criteria: str = "", excerpt: str = ""):
#         self.chain = chain
#         self.required_keys = required_keys
#         self.criteria = criteria
#         self.excerpt = excerpt
#         if validity_fn is None:
#             self.validity_fn = lambda x: bool(isinstance(x, dict))
        
#         assert self.validity_fn({}), "Empty dictionary failed."
#         # assert self.validity_fn('{}'), "String failed."
    
#     def is_output_valid(self, output_object: dict) -> bool:
#         """
#         Checks if the output is valid.
#         """
#         for key in self.required_keys:
#             if key not in output_object:
#                 return False
#         result = output_object["meetsCriteria"]
#         difficulty = output_object["difficulty"]
#         assert isinstance(result, bool)
#         assert isinstance(difficulty, str) and difficulty in ["easy", "hard"]
#         print("Meets criteria:", result)
#         return True
    
#     def test_chain(self):
#         assert all([self.excerpt, self.criteria])

#         res = self.chain.invoke({"excerpt": self.excerpt, "criteria": self.criteria})
#         print(res)

class TestChains(unittest.TestCase):
    from pyperclip import paste
    def setUp(self):
        UNLOADED_MODEL = LLM_FN(MODEL_FN)
        self.llm = (LLM(UNLOADED_MODEL)).llm
    
    def test_eval_chain(self):
        def is_output_valid(output_object: dict) -> bool:
            """
            Checks if the output is valid.
            """
            required_keys = ["difficulty", "meetsCriteria", "reasoning"]
            for key in required_keys:
                if key not in output_object:
                    return False
            result = output_object["meetsCriteria"]
            difficulty = output_object["difficulty"]
            assert isinstance(result, bool)
            assert isinstance(difficulty, str) and difficulty in ["easy", "hard"]
            print("Meets criteria:", result)
            return True
        chain = get_eval_chain(self.llm, is_output_valid)
        # EXCERPT = "130-21-1221"
        # CRITERIA = "Evaluating the expression gets a result that is odd."
        EXCERPT = "reformatter.py requirements.txt response.md routing.py run_server.py"
        CRITERIA = "This is likely to have something to do with nuclear tech."
        criteria_suffix = ' Use 3 keys: reasoning, meetsCriteria, difficulty ("easy" or "hard").'
        eval_inputs = {"excerpt": EXCERPT, "criteria": CRITERIA + criteria_suffix}
        res = chain.invoke(eval_inputs)
        print(res)

if __name__ == "__main__":
    # llm = LLM(LLM_FN(MODEL_FN)).llm
    # required_keys = ["reasoning", "meetsCriteria"]
    # test = EvalTest(get_eval_chain(llm, lambda x: True), required_keys=required_keys, excerpt="130-21-1221", criteria="Evaluating the expression gets a result that is odd. Add a reasoning key.")
    # test.validity_fn = test.is_output_valid
    # test.test_chain()
    unittest.main()
    

# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")


# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_current_weather",
#             "description": "Get the current weather",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state, e.g. San Francisco, CA",
#                     },
#                     "format": {
#                         "type": "string",
#                         "enum": ["celsius", "fahrenheit"],
#                         "description": "The temperature unit to use. Infer this from the users location.",
#                     },
#                 },
#                 "required": ["location", "format"],
#             },
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "get_n_day_weather_forecast",
#             "description": "Get an N-day weather forecast",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state, e.g. San Francisco, CA",
#                     },
#                     "format": {
#                         "type": "string",
#                         "enum": ["celsius", "fahrenheit"],
#                         "description": "The temperature unit to use. Infer this from the users location.",
#                     },
#                     "num_days": {
#                         "type": "integer",
#                         "description": "The number of days to forecast",
#                     }
#                 },
#                 "required": ["location", "format", "num_days"]
#             },
#         }
#     },
# ]


# def test_models():
#     llm = None
#     if Test_OpenAI:
#         llm = LLM(LLM_FN(get_openai_gpt4))
#         if not ENCODE_FROM_URL:
#             filetype = IMAGE_PATH.split(".")[-1]
#             base64_image = encode_image(IMAGE_PATH)
#             image_url = f"data:image/{filetype};base64,{base64_image}"
#         else:
#             image_url = IMAGE_URL
#         image_dict = {"url": image_url}
#         messages = [
#             SystemMessage(content="You are a helpful AI model that analyzes images."),
#             HumanMessage(content=[
#                 {"type": "text", "text": "What is this an image of? Does the image have any red color?"},
#                 {"type": "image_url", "image_url": image_dict}
#             ])
#         ]
#         response_string = ""
#         for chunk in llm.stream(messages):
#             response_string += chunk.content
#             print(chunk.content, end="")
#         print("\nLength of response:", len(response_string))
#     if Test_Anthropic:
#         llm = LLM(LLM_FN(get_claude_opus))
#     if Test_Together:
#         llm = LLM(LLM_FN(get_together_llama3))
#     if Test_LMStudio:
#         llm = LLM(LLM_FN(get_local_model))
#     if Test_Ollama:
#         llm = LLM(LLM_FN(get_ollama_llama3))
#         llm = LLM(LLM_FN(get_ollama_mistral))
#     if llm:
#         llm = llm.llm
#     print()
    # print("All models are working correctly.")
# test_models()
