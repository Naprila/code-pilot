from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from codepilot.llm.factory import get_llm
from codepilot.agent.tools import search_codebase
from codepilot.observability.logger import get_logger
from codepilot.memory.short_term import get_checkpointer, get_summarization_middleware

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a senior software engineer with deep knowledge of the codebase.
Use search_codebase only when the user asks about code concepts, symbols, or behaviors that require repository lookup.
Do not call tools repeatedly for the same question. Once you have enough evidence, answer directly.
Reference specific file names, function names and line numbers in your answers.
If you cannot find the answer in the codebase, say so explicitly."""

def build_agent():
    """Create and return a LangChain agent with persistent memory."""
    llm = get_llm()
    tools = [search_codebase]
    checkpointer = get_checkpointer()
    middleware = get_summarization_middleware()
    logger.info("Creating agent")
    
    return create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[middleware],
    )