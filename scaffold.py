# scaffold.py
import argparse
import os
import re

# --- Boilerplate File Templates ---

FILE_TEMPLATES = {
    "config.py": """
# config.py
\"\"\"Global configuration constants for the application.\"\"\"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-2.0-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{DEFAULT_MODEL}:generateContent"
""",
    "main.py": """
# main.py
import sys
from orchestrator.text_agent_graph_loader import TextAgentGraphLoader
from utils.logger import logger

def main():
    \"\"\"The main entry point for the text-based multi-agent system.\"\"\"
    logger.info("Starting Text-to-Text Multi-Agent System...")

    try:
        orchestrator = TextAgentGraphLoader.load_from_config(
            workflows_path="text_agent_workflows.json",
            agents_config_path="text_agents_config.json"
        )
    except Exception as e:
        logger.error(f"Failed to initialize the system: {e}")
        sys.exit(1)

    print("\\n--- Text Agent System Initialized ---")
    print("Enter 'exit' to quit the application.")
    
    while True:
        try:
            print(f"\\nCurrent Agent: {orchestrator.active_agent_name}")
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            
            response = orchestrator.process_user_message(user_input)
            print(f"\\nAgent ({orchestrator.active_agent_name}): {response}")

        except KeyboardInterrupt:
            print("\\nExiting...")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            print("An unexpected error occurred. Please check the logs.")

if __name__ == "__main__":
    main()
""",
    "text_agents_config.json": """
{
  "tools": {
    "sample_tool": {
      "class": "SampleTool",
      "module": "tools.sample_tool"
    }
  }
}
""",
    "text_agent_workflows.json": """
{
  "agent_graph": {
    "start_node": "GreetingAgent",
    "agents": {
      "GreetingAgent": {
        "type": "conversational",
        "initial_prompt": "You are a friendly assistant. Greet the user and ask how you can help. If they ask for help, transition to the HelperAgent.",
        "transitions": {
          "user needs help": "HelperAgent"
        }
      },
      "HelperAgent": {
        "type": "tool_execution",
        "initial_prompt": "You are a helpful assistant. Use the 'sample_tool' to help the user.",
        "tools": ["sample_tool"],
        "transitions": {
          "task is complete": "GreetingAgent"
        }
      }
    }
  }
}
""",
    "agents/base_text_agent.py": """
# agents/base_text_agent.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from utils.data_models import Message
from utils.workflow_state import WorkflowState

class BaseTextAgent(ABC):
    \"\"\"Abstract Base Class for all text-based agents.\"\"\"
    def __init__(self, agent_name: str, system_prompt: str, transitions: Dict[str, str]):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.transitions = transitions

    @abstractmethod
    def process_text_input(
        self, 
        user_input: str, 
        chat_history: List[Message],
        workflow_state: Optional[WorkflowState] = None
    ) -> str:
        pass
""",
    "agents/text_conversational_agent.py": """
# agents/text_conversational_agent.py
from typing import List, Dict, Optional
from agents.base_text_agent import BaseTextAgent
from llm_clients.base_prediction_client import BasePredictionClient
from utils.data_models import Message
from utils.workflow_state import WorkflowState
from utils.logger import logger

class TextConversationalAgent(BaseTextAgent):
    def __init__(self, agent_name: str, system_prompt: str, llm_client: BasePredictionClient, transitions: Dict[str, str]):
        super().__init__(agent_name, system_prompt, transitions)
        self.llm_client = llm_client

    def process_text_input(self, user_input: str, chat_history: List[Message], workflow_state: Optional[WorkflowState] = None) -> str:
        logger.info(f"[{self.agent_name}] Processing input...")
        messages = [{"role": msg.role, "parts": [{"text": msg.content}]} for msg in chat_history]
        messages.append({"role": "user", "parts": [{"text": user_input}]})
        response = self.llm_client.get_text_completion(messages=messages, system_prompt=self.system_prompt)
        return response.text or "Sorry, I could not generate a response."
""",
    "agents/text_tool_execution_agent.py": """
# agents/text_tool_execution_agent.py
from typing import List, Dict, Optional
from agents.base_text_agent import BaseTextAgent
from llm_clients.base_prediction_client import BasePredictionClient
from tools.base_tool import BaseTool
from utils.data_models import Message
from utils.workflow_state import WorkflowState
from utils.logger import logger

class TextToolExecutionAgent(BaseTextAgent):
    def __init__(self, agent_name: str, system_prompt: str, llm_client: BasePredictionClient, tools: List[BaseTool], transitions: Dict[str, str]):
        super().__init__(agent_name, system_prompt, transitions)
        self.llm_client = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.tool_definitions = [tool.get_definition() for tool in tools]

    def process_text_input(self, user_input: str, chat_history: List[Message], workflow_state: Optional[WorkflowState] = None) -> str:
        messages = [{"role": msg.role, "parts": [{"text": msg.content}]} for msg in chat_history]
        messages.append({"role": "user", "parts": [{"text": user_input}]})

        llm_response = self.llm_client.get_text_completion(messages=messages, system_prompt=self.system_prompt, tools=self.tool_definitions)

        if llm_response.tool_calls:
            tool_call = llm_response.tool_calls[0]
            if tool_call.name in self.tools:
                tool = self.tools[tool_call.name]
                tool_result = tool.run(state=workflow_state, **tool_call.args)
                
                messages.append({"role": "model", "parts": [{"functionCall": {"name": tool_call.name, "args": tool_call.args}}]})
                messages.append({"role": "function", "parts": [{"functionResponse": {"name": tool_call.name, "response": {"content": tool_result}}}]})
                
                summary_response = self.llm_client.get_text_completion(messages=messages, system_prompt=self.system_prompt)
                return summary_response.text or "Tool executed. Ready for next step."
        
        return llm_response.text or "I'm not sure how to respond."
""",
    "llm_clients/base_prediction_client.py": """
# llm_clients/base_prediction_client.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from utils.data_models import LLMResponse

class BasePredictionClient(ABC):
    @abstractmethod
    def get_text_completion(self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None, tools: List[Dict[str, Any]] = None) -> LLMResponse:
        pass
""",
    "llm_clients/gemini_prediction_client.py": """
# llm_clients/gemini_prediction_client.py
import requests
import json
from typing import List, Dict, Any, Optional
from config import GEMINI_API_KEY, GEMINI_API_URL
from utils.logger import logger
from utils.data_models import LLMResponse, ToolCall
from llm_clients.base_prediction_client import BasePredictionClient

class GeminiPredictionClient(BasePredictionClient):
    def get_text_completion(self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None, tools: List[Dict[str, Any]] = None) -> LLMResponse:
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
            raise ValueError("Gemini API key is not configured in config.py.")

        headers = {"Content-Type": "application/json"}
        payload = {"contents": messages, "generationConfig": {"temperature": 0.7}}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if tools:
            payload["tools"] = [{"function_declarations": tools}]
        
        params = {"key": GEMINI_API_KEY}
        try:
            response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload, timeout=120)
            response.raise_for_status()
            response_data = response.json()
            
            candidate = response_data.get("candidates", [{}])[0]
            first_part = candidate.get("content", {}).get("parts", [{}])[0]

            if "functionCall" in first_part:
                tool_call_data = first_part["functionCall"]
                return LLMResponse(text=None, tool_calls=[ToolCall(name=tool_call_data["name"], args=tool_call_data.get("args", {}))])
            
            return LLMResponse(text=first_part.get("text", ""), tool_calls=None)
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling Gemini API: {e}")
            return LLMResponse(text="Error: Network error.", tool_calls=None)
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing Gemini API response: {e}")
            return LLMResponse(text="Error: Invalid API response.", tool_calls=None)
""",
    "orchestrator/text_agent_orchestrator.py": """
# orchestrator/text_agent_orchestrator.py
import re
from typing import Dict, List
from agents.base_text_agent import BaseTextAgent
from utils.data_models import Message
from utils.logger import logger

class TextAgentOrchestrator:
    def __init__(self, agents: Dict[str, BaseTextAgent], start_node: str):
        self.agents = agents
        self.active_agent_name = start_node
        self.chat_history: List[Message] = []

    @property
    def active_agent(self) -> BaseTextAgent:
        return self.agents[self.active_agent_name]

    def _check_for_transition(self, response_text: str) -> bool:
        if "TRANSITION_TO:" in response_text:
            match = re.search(r"TRANSITION_TO:\s*(\\w+)", response_text)
            if match:
                target_agent_name = match.group(1).strip()
                if target_agent_name in self.agents:
                    self.active_agent_name = target_agent_name
                    logger.info(f"Transitioning to agent: {target_agent_name}")
                    return True
        return False

    def process_user_message(self, message: str) -> str:
        self.chat_history.append(Message(role="user", content=message))
        agent_response = self.active_agent.process_text_input(message, self.chat_history)
        
        if not self._check_for_transition(agent_response):
            self.chat_history.append(Message(role="model", content=agent_response))
        
        self.chat_history = self.chat_history[-10:]
        return agent_response
""",
    "orchestrator/text_agent_graph_loader.py": """
# orchestrator/text_agent_graph_loader.py
import json
import importlib
from agents.text_conversational_agent import TextConversationalAgent
from agents.text_tool_execution_agent import TextToolExecutionAgent
from llm_clients.gemini_prediction_client import GeminiPredictionClient
from orchestrator.text_agent_orchestrator import TextAgentOrchestrator

class TextAgentGraphLoader:
    @staticmethod
    def load_from_config(workflows_path: str, agents_config_path: str) -> TextAgentOrchestrator:
        with open(workflows_path, 'r') as f:
            workflows = json.load(f)
        with open(agents_config_path, 'r') as f:
            agents_config = json.load(f)

        llm_client = GeminiPredictionClient()
        
        available_tools = {}
        for name, info in agents_config["tools"].items():
            module = importlib.import_module(info["module"])
            tool_class = getattr(module, info["class"])
            available_tools[name] = tool_class()

        agents = {}
        for name, info in workflows["agent_graph"]["agents"].items():
            tools = [available_tools[t] for t in info.get("tools", [])]
            if info["type"] == "conversational":
                agents[name] = TextConversationalAgent(name, info["initial_prompt"], llm_client, info["transitions"])
            elif info["type"] == "tool_execution":
                agents[name] = TextToolExecutionAgent(name, info["initial_prompt"], llm_client, tools, info["transitions"])
        
        return TextAgentOrchestrator(agents, workflows["agent_graph"]["start_node"])
""",
    "tools/base_tool.py": """
# tools/base_tool.py
from abc import ABC, abstractmethod
from typing import Dict, Optional
from utils.workflow_state import WorkflowState

class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: pass
    @property
    @abstractmethod
    def description(self) -> str: pass
    @property
    @abstractmethod
    def parameters(self) -> Dict: pass
    @abstractmethod
    def run(self, state: Optional[WorkflowState] = None, **kwargs) -> str: pass
    def get_definition(self) -> Dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}
""",
    "tools/sample_tool.py": """
# tools/sample_tool.py
from tools.base_tool import BaseTool
from typing import Dict, Optional
from utils.workflow_state import WorkflowState

class SampleTool(BaseTool):
    @property
    def name(self) -> str: return "sample_tool"
    @property
    def description(self) -> str: return "A sample tool that takes a string and returns it in uppercase."
    @property
    def parameters(self) -> Dict:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    def run(self, state: Optional[WorkflowState] = None, **kwargs) -> str:
        text = kwargs.get("text", "")
        return text.upper()
""",
    "utils/data_models.py": """
# utils/data_models.py
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Message:
    role: str
    content: str

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]

@dataclass
class LLMResponse:
    text: str | None
    tool_calls: List[ToolCall] | None
""",
    "utils/logger.py": """
# utils/logger.py
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger("LLMAgentFramework")
""",
    "utils/workflow_state.py": """
# utils/workflow_state.py
from typing import Dict, Any

class WorkflowState:
    def __init__(self, initial_params: Dict[str, Any]):
        self._state: Dict[str, Any] = initial_params
    def set(self, key: str, value: Any): self._state[key] = value
    def get(self, key: str, default: Any = None) -> Any: return self._state.get(key, default)
    def get_all(self) -> Dict[str, Any]: return self._state
"""
}

# --- Individual Component Templates (for the 'new' command) ---

TOOL_TEMPLATE = FILE_TEMPLATES["tools/sample_tool.py"] # Reuse for consistency
AGENT_TEMPLATE = """
# agents/{agent_name}_agent.py
# This is a generic agent. You will need to decide if it's a 
# TextConversationalAgent or TextToolExecutionAgent and add the
# appropriate imports and class inheritance.
from agents.base_text_agent import BaseTextAgent
from typing import List, Dict, Optional
from utils.data_models import Message
from utils.workflow_state import WorkflowState
from utils.logger import logger

class {class_name}Agent(BaseTextAgent): # <-- CHANGE THIS BASE CLASS
    def __init__(self, agent_name: str, system_prompt: str, llm_client, transitions: Dict[str, str], tools: List = []):
        super().__init__(agent_name, system_prompt, transitions)
        self.llm_client = llm_client

    def process_text_input(
        self, 
        user_input: str, 
        chat_history: List[Message],
        workflow_state: Optional[WorkflowState] = None
    ) -> str:
        logger.info(f"[{self.agent_name}] is processing the input...")
        return "This is a response from the newly created {class_name}Agent."
"""

# --- Helper Functions ---

def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def to_pascal_case(name):
    return ''.join(word.capitalize() for word in re.split('_|-', name))

# --- Scaffolding Logic ---

def init_project(args):
    """Creates the entire boilerplate project structure."""
    project_name = args.name
    if os.path.exists(project_name):
        print(f"❌ Error: Directory '{project_name}' already exists.")
        return

    print(f"Initializing new LLM Agent project: {project_name}")
    os.makedirs(project_name)

    for filepath, content in FILE_TEMPLATES.items():
        full_path = os.path.join(project_name, filepath)
        dir_name = os.path.dirname(full_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        with open(full_path, "w") as f:
            f.write(content.strip())
        print(f"  ✅ Created {full_path}")

    print("\n🚀 Project initialization complete!")
    print("\n--- Next Steps ---")
    print(f"1.  Navigate into your new project: `cd {project_name}`")
    print("2.  Set your API key in `config.py`.")
    print("3.  Run the project: `python main.py`")

def create_tool(args):
    """Generates a new tool file from the template."""
    name = args.name
    class_name = to_pascal_case(name) + "Tool"
    tool_name = to_snake_case(name)
    filepath = os.path.join("tools", f"{tool_name}_tool.py")
    
    # ... (rest of the create_tool logic remains the same)
    print(f"Creating new tool: {class_name}...")
    
    os.makedirs("tools", exist_ok=True)

    if os.path.exists(filepath):
        print(f"❌ Error: File already exists at '{filepath}'")
        return

    content = TOOL_TEMPLATE.replace("SampleTool", class_name).replace("sample_tool", tool_name)
    with open(filepath, "w") as f:
        f.write(content.strip())

    print(f"✅ Success! Tool created at '{filepath}'")


def create_agent(args):
    """Generates a new agent file from the template."""
    name = args.name
    class_name = to_pascal_case(name)
    agent_name_snake = to_snake_case(name)
    filepath = os.path.join("agents", f"{agent_name_snake}_agent.py")

    # ... (rest of the create_agent logic remains the same)
    print(f"Creating new agent: {class_name}Agent...")

    os.makedirs("agents", exist_ok=True)
    
    if os.path.exists(filepath):
        print(f"❌ Error: File already exists at '{filepath}'")
        return

    content = AGENT_TEMPLATE.format(class_name=class_name, agent_name=agent_name_snake)
    with open(filepath, "w") as f:
        f.write(content.strip())
    
    print(f"✅ Success! Agent created at '{filepath}'")


def main():
    """The main entry point for the command-line tool."""
    parser = argparse.ArgumentParser(description="Scaffolding tool for the LLM Agent Framework.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- 'init' command ---
    init_parser = subparsers.add_parser("init", help="Initialize a new agent framework project.")
    init_parser.add_argument("--name", type=str, required=True, help="The name of the new project directory.")
    init_parser.set_defaults(func=init_project)

    # --- 'new' command ---
    new_parser = subparsers.add_parser("new", help="Create a new component inside an existing project.")
    new_subparsers = new_parser.add_subparsers(dest="type", required=True)

    tool_parser = new_subparsers.add_parser("tool", help="Create a new tool.")
    tool_parser.add_argument("--name", type=str, required=True, help="The name of the tool (e.g., 'FileParser').")
    tool_parser.set_defaults(func=create_tool)

    agent_parser = new_subparsers.add_parser("agent", help="Create a new agent.")
    agent_parser.add_argument("--name", type=str, required=True, help="The name of the agent (e.g., 'DataAnalyzer').")
    agent_parser.set_defaults(func=create_agent)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
