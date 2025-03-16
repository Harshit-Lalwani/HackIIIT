from moya.agents.openai_agent import OpenAIAgent, OpenAIAgentConfig
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI credentials
azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

# Define Azure deployment name - you'll need to replace this with your actual deployment name
deployment_name = "gpt-4o"  # Replace with your Azure deployment name

# Create an Azure OpenAI client
azure_client = AzureOpenAI(
    api_key=azure_api_key,
    api_version=azure_api_version,
    azure_endpoint=azure_endpoint
)

# Create an agent configuration
agent_config = OpenAIAgentConfig(
    agent_name="my_assistant",
    description="A general-purpose AI assistant",
    system_prompt="You are a helpful AI assistant.",
    agent_type = "OpenAIAgent",
    model_name=deployment_name,  # This should match your Azure deployment name
    api_key=azure_api_key  # This will be used for authentication
)

# Initialize the agent
agent = OpenAIAgent(config=agent_config)

# Replace the default OpenAI client with Azure OpenAI client
agent.client = azure_client

# Send a message and get a response
try:
    response = agent.handle_message("Hello, how can you help me?")
    print(response)
except Exception as e:
    print(f"Error: {e}")
    # For debugging
    print(f"Details about agent configuration:")
    print(f"Model name: {agent.model_name}")
    print(f"Azure endpoint: {azure_endpoint}")
    print(f"API version: {azure_api_version}")