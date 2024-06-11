from pydantic import BaseModel
from ray import serve
from starlette.requests import Request
from models import get_local_model
from rag import get_eval_chain
DEFAULT_QUERY = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
DEFAULT_CRITERIA = "The language is Spanish."

REQUIRE_CRITERIA = False
# REQUIRED PARAMS: text, criteria
class ChainInputs(BaseModel):
    excerpt: str
    criteria: str


# 1: Define a Ray Serve deployment.
# @serve.deployment
class DeployLLM:
    def __init__(self):
        # We initialize the LLM, template and the chain here
        llm = get_local_model()
        self.chain = get_eval_chain(llm)

    def _run_chain(self, inputs: ChainInputs):
        inputs_dict = {
            "excerpt": inputs.excerpt,
            "criteria": inputs.criteria,
        }
        return self.chain.invoke(inputs_dict)

    async def __call__(self, request: Request):
        # 1. Parse the request
        text = request.query_params.get("text", "").strip()
        if not text:
            return "Please provide a text parameter."
        criteria = request.query_params.get("criteria", "").strip()
        if not criteria:
            if REQUIRE_CRITERIA:
                return "Please provide a criteria parameter."
            print("Using default criteria")
            criteria = DEFAULT_CRITERIA
        
        inputs = ChainInputs(excerpt=text, criteria=criteria)
        # 2. Run the chain
        resp = self._run_chain(inputs)
        # 3. Return the response
        return resp

# Bind the model to deployment
# deployment = DeployLLM.bind()

inputs = ChainInputs(excerpt=DEFAULT_QUERY, criteria=DEFAULT_CRITERIA)
response = DeployLLM()._run_chain(inputs)
print(response)