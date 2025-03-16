"""
Example of a mother agent creating and interacting with a child agent.
"""

import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from moya.agents.openai_agent import OpenAIAgent, OpenAIAgentConfig
from moya.registry.agent_registry import AgentRegistry
from moya.tools.tool_registry import ToolRegistry
from moya.tools.azure_agent_creator import AzureAgentCreator
from moya.tools.base_tool import BaseTool

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI credentials
azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
deployment_name = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o")

# Initialize registries
agent_registry = AgentRegistry()
tool_registry = ToolRegistry()

# Add list_functions method to BaseTool if it doesn't exist
if not hasattr(BaseTool, 'list_functions'):
    def list_functions(self):
        """Return a list of function definitions for OpenAI format"""
        functions = []
        tool_defs = self.get_tool_definitions() if hasattr(self, 'get_tool_definitions') else {}
        
        for method_name, method_def in tool_defs.items():
            functions.append({
                "name": method_name,
                "description": method_def.get("description", ""),
                "parameters": method_def.get("parameters", {})
            })
        return functions
    
    BaseTool.list_functions = list_functions

# Register the Azure Agent Creator tool
creator_tool = AzureAgentCreator(agent_registry=agent_registry)
tool_registry.register_tool(creator_tool)

# Modify OpenAIAgent to properly handle tools
def get_tool_definitions_patched(self):
    """Patched version of get_tool_definitions that uses list_functions"""
    if not hasattr(self, 'config') or not hasattr(self.config, 'tool_registry'):
        return []
        
    tool_definitions = []
    for tool in self.config.tool_registry.get_tools():
        if hasattr(tool, 'list_functions'):
            tool_functions = tool.list_functions()
            if tool_functions:
                tool_definitions.extend(tool_functions)
    
    return tool_definitions

# Apply the patch to OpenAIAgent
OpenAIAgent.get_tool_definitions = get_tool_definitions_patched

# Create the mother agent with tool calling capability
mother_agent_config = OpenAIAgentConfig(
    agent_name="mother_agent",
    description="A mother agent that can create child agents",
    system_prompt=(
        "You are a Mother Agent with the ability to design and create child agents. "
        "Your goal is to design the most helpful and effective child agents based on requirements. "
        "When asked to create a child agent, think carefully about the ideal system prompt, "
        "personality, and capabilities for that agent. "
        "You have access to the 'create_azure_agent' function which allows you to create new agents."
    ),
    agent_type="OpenAIAgent",
    model_name=deployment_name,
    api_key=azure_api_key,
    tool_registry=tool_registry,
    is_tool_caller=True
)

# Initialize the mother agent
mother_agent = OpenAIAgent(config=mother_agent_config)

# Replace the default OpenAI client with Azure OpenAI client
azure_client = AzureOpenAI(
    api_key=azure_api_key,
    api_version=azure_api_version,
    azure_endpoint=azure_endpoint
)
mother_agent.client = azure_client

# Ask the mother agent to create a child agent
design_request = (
    "Please design an ideal child agent that specializes in education and helping students. "
    "This agent should be warm, supportive, and knowledgeable about learning methods. "
    "Use the create_azure_agent function to create this child agent with an "
    "appropriate system prompt that defines its personality and capabilities. "
    "For the function parameters, use these values:\n"
    "- endpoint: " + azure_endpoint + "\n"
    "- api_key: " + azure_api_key + "\n"
    "- api_version: " + azure_api_version + "\n"
    "- deployment_name: " + deployment_name + "\n"
)

print("Asking mother agent to design an educational child agent...")
response = mother_agent.handle_message(design_request)
print("\nMother Agent's Response:")
print(response)

# Retrieve the newly created child agent
child_agent = None
# Use list_agents() method - based on the error, this is the correct way to access agents
agent_list = agent_registry.list_agents()

print(f"\nAgents in registry: {len(agent_list)}")
for agent_info in agent_list:
    print(f"Found agent: {agent_info.name}")
    # Skip the mother agent
    if agent_info.name != "mother_agent":
        # Get the agent using the get_agent method
        child_agent = agent_registry.get_agent(agent_info.name)
        break

if child_agent:
    print("\nInteracting with the created child agent...")
    child_response = child_agent.handle_message("Who are you?")
    print("\nChild Agent's Response to 'Who are you?':")
    print(child_response)
else:
    print("\nNo child agent was created or found in the registry.")