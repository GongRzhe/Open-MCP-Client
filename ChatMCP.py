try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
import asyncio
import json
import logging
import os
import random
import shutil
import subprocess
import time
from copy import deepcopy
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Literal, Union, Tuple, Callable, TypeVar, Awaitable

import aiohttp  
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pydantic

# Configure logging once
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='chatbot.log',
    filemode='a'
)
class ToolLogFilter(logging.Filter):
    def filter(self, record):
        return "Executing tool" in record.getMessage() or "tool execution" in record.getMessage()

tool_console = logging.StreamHandler()
tool_console.setLevel(logging.INFO)
tool_console.setFormatter(logging.Formatter('Tool: %(message)s'))
tool_console.addFilter(ToolLogFilter())
logging.getLogger().addHandler(tool_console)
# # Add console handler only for critical errors
# console = logging.StreamHandler()
# console.setLevel(logging.ERROR)  # Only show ERROR and CRITICAL
# formatter = logging.Formatter('%(levelname)s: %(message)s')
# console.setFormatter(formatter)
# logging.getLogger().addHandler(console)
class TokenUsage:
    """Tracks token usage across different providers."""
    
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
    
    def update(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.prompt_tokens = prompt_tokens or self.prompt_tokens
        self.completion_tokens = completion_tokens or self.completion_tokens
        self.total_tokens = total_tokens or (self.prompt_tokens + self.completion_tokens)
    
    def __str__(self):
        return f"[Tokens: {self.prompt_tokens} in, {self.completion_tokens} out, {self.total_tokens} total]"

class ServerState(Enum):
    """Enum representing server connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


class CircuitBreakerState(Enum):
    """Enum representing circuit breaker states."""
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


T = TypeVar('T')


class CircuitBreaker:
    """Implements the Circuit Breaker pattern to prevent cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_time: int = 30) -> None:
        """Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_time: Time in seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()
    
    async def execute(self, coro_func: Callable[[], Awaitable[T]]) -> T:
        """Execute the coroutine function if the circuit is not open.
        
        Args:
            coro_func: Callable that returns a coroutine to execute
            
        Returns:
            Result of the coroutine
            
        Raises:
            RuntimeError: If the circuit is open
        """
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                # Check if recovery time has elapsed
                if time.time() - self.last_failure_time >= self.recovery_time:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logging.info("Circuit half-open, attempting recovery...")
                else:
                    raise RuntimeError("Circuit breaker is open")
            
            try:
                result = await coro_func()
                
                # Reset failure count on success
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logging.info("Circuit closed, service recovered successfully")
                
                return result
            
            except Exception as e:
                # Update failure stats
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Open circuit if threshold reached
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logging.warning(f"Circuit opened after {self.failure_count} failures")
                
                raise e


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        # API keys for different LLM providers
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")  # Consistent naming with provider
        self.openroute_api_key = os.getenv("OPENROUTE_API_KEY")
        # self.ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.ollama_host = "http://127.0.0.1:11434"
        self.github_api_key = os.getenv("GITHUB_API_KEY")
        
        # Default LLM provider and model
        self.default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "groq")
        self.default_model = os.getenv("DEFAULT_LLM_MODEL", "llama-3.2-90b-vision-preview")
        
        # Connection settings with safe parsing
        self.connection_timeout = self._safe_parse_int("CONNECTION_TIMEOUT", 10)
        self.read_timeout = self._safe_parse_int("READ_TIMEOUT", 60)
        self.max_retries = self._safe_parse_int("MAX_RETRIES", 3)
        self.retry_delay_base = self._safe_parse_float("RETRY_DELAY_BASE", 1.0)
        self.retry_max_delay = self._safe_parse_float("RETRY_MAX_DELAY", 30.0)
        self.health_check_interval = self._safe_parse_int("HEALTH_CHECK_INTERVAL", 60)
        self.message_history_limit = self._safe_parse_int("MESSAGE_HISTORY_LIMIT", 20)
        self.server_init_timeout = self._safe_parse_int("SERVER_INIT_TIMEOUT", 30)  # 30 seconds default

    def _safe_parse_int(self, env_var: str, default: int) -> int:
        """Safely parse an integer environment variable.
        
        Args:
            env_var: Name of the environment variable
            default: Default value if parsing fails
            
        Returns:
            Parsed integer value or default
        """
        try:
            value = os.getenv(env_var)
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            logging.warning(f"Invalid value for {env_var}, using default: {default}")
            return default
            
    def _safe_parse_float(self, env_var: str, default: float) -> float:
        """Safely parse a float environment variable.
        
        Args:
            env_var: Name of the environment variable
            default: Default value if parsing fails
            
        Returns:
            Parsed float value or default
        """
        try:
            value = os.getenv(env_var)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            logging.warning(f"Invalid value for {env_var}, using default: {default}")
            return default

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file.
        
        Args:
            file_path: Path to the JSON configuration file.
            
        Returns:
            Dict containing server configuration.
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
                
            # Validate config has required fields
            if "mcpServers" not in config:
                raise ConfigurationError("Configuration missing 'mcpServers' section")
                
            return config
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {file_path}")

    def get_api_key(self, provider: str) -> str:
        """Get the API key for the specified provider.
        
        Args:
            provider: The LLM provider name.
            
        Returns:
            The API key as a string.
            
        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        api_key_map = {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,  # Consistent naming
            "openroute": self.openroute_api_key,
            "github": self.github_api_key,
        }
        
        api_key = api_key_map.get(provider.lower())
        if not api_key and provider.lower() != "ollama":
            raise ValueError(f"{provider.upper()}_API_KEY not found in environment variables")
        return api_key


class AsyncRetry:
    """Handles asynchronous retry logic with exponential backoff and jitter."""
    
    def __init__(
        self, 
        max_retries: int, 
        base_delay: float, 
        max_delay: float,
        jitter: bool = True
    ) -> None:
        """Initialize retry parameters.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            jitter: Whether to add randomness to delay to prevent thundering herd
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
    
    async def execute(self, coro_func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute a coroutine with retries.
        
        Args:
            coro_func: Coroutine function to retry
            *args: Arguments to pass to the coroutine
            **kwargs: Keyword arguments to pass to the coroutine
            
        Returns:
            Result from the coroutine
            
        Raises:
            Exception: Last exception raised after all retries fail
        """
        attempt = 0
        last_exception = None
        
        while attempt <= self.max_retries:
            try:
                if attempt > 0:
                    logging.info(f"Retry attempt {attempt}/{self.max_retries}")
                    
                return await coro_func(*args, **kwargs)
                
            except Exception as e:
                attempt += 1
                last_exception = e
                
                if attempt > self.max_retries:
                    logging.error(f"Max retries ({self.max_retries}) exceeded")
                    raise
                
                # Calculate exponential backoff delay
                delay = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
                
                # Add jitter if enabled
                if self.jitter:
                    delay = delay * (0.5 + random.random())
                
                logging.warning(f"Operation failed: {str(e)}. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
        
        # This should never happen but just in case
        raise last_exception if last_exception else RuntimeError("Retry failed for unknown reason")


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any], global_config: Configuration) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.global_config: Configuration = global_config
        self.stdio_context: Optional[Any] = None
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.capabilities: Optional[Dict[str, Any]] = None
        self.state: ServerState = ServerState.DISCONNECTED
        self.last_connection_attempt: float = 0
        self.connection_failures: int = 0
        self.max_consecutive_failures: int = 3
        self.reconnect_delay: float = 5.0
        self.tools: List[Any] = []
        self.resources: List[Any] = []  # Added to cache resources
        self.prompts: List[Any] = []    # Added to cache prompts
        self.circuit_breaker = CircuitBreaker()
        self.retry_handler = AsyncRetry(
            max_retries=global_config.max_retries,
            base_delay=global_config.retry_delay_base,
            max_delay=global_config.retry_max_delay
        )
        self._health_check_task: Optional[asyncio.Task] = None

    async def list_resources(self) -> List[Any]:
        """List available resources from the server with reconnection logic.
        
        Returns:
            A list of available resources.
        """
        # If disconnected, try to reconnect
        if self.state != ServerState.CONNECTED:
            if time.time() - self.last_connection_attempt > self.reconnect_delay:
                reconnected = await self.initialize()
                if not reconnected:
                    return []
            else:
                return []
        
        # Return cached resources list to avoid frequent server calls
        return self.resources

    async def list_prompts(self) -> List[Any]:
        """List available prompts from the server with reconnection logic.
        
        Returns:
            A list of available prompts.
        """
        # If disconnected, try to reconnect
        if self.state != ServerState.CONNECTED:
            if time.time() - self.last_connection_attempt > self.reconnect_delay:
                reconnected = await self.initialize()
                if not reconnected:
                    return []
            else:
                return []
        
        # Return cached prompts list to avoid frequent server calls
        return self.prompts

    async def read_resource(self, uri: str) -> Any:
        """Read a resource by URI with retry logic.
        
        Args:
            uri: The URI of the resource to read.
            
        Returns:
            Resource content.
            
        Raises:
            RuntimeError: If resource reading fails.
        """
        # If not connected, try to reconnect
        if self.state != ServerState.CONNECTED:
            reconnected = await self.initialize()
            if not reconnected:
                raise RuntimeError(f"Cannot read resource: server {self.name} is not connected")
        
        # Use circuit breaker pattern to prevent cascading failures
        try:
            return await self.circuit_breaker.execute(
                lambda: self.retry_handler.execute(self._read_resource_internal, uri)
            )
        except Exception as e:
            logging.error(f"Failed to read resource {uri} after retries: {e}")
            self.state = ServerState.FAILED
            await self.cleanup()
            self.last_connection_attempt = time.time()
            raise RuntimeError(f"Resource reading failed: {str(e)}")

    async def _read_resource_internal(self, uri: str) -> Any:
        """Internal resource reading function wrapped by retry handler.
        
        Args:
            uri: The URI of the resource to read.
            
        Returns:
            Resource content.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        
        logging.info(f"Reading resource {uri}...")
        result = await self.session.read_resource(uri)
        return result

    async def get_prompt(self, name: str, arguments: Dict[str, Any] = None) -> Any:
        """Get a prompt template with optional arguments.
        
        Args:
            name: The name of the prompt.
            arguments: Optional arguments for the prompt.
            
        Returns:
            Prompt result with messages.
            
        Raises:
            RuntimeError: If prompt execution fails.
        """
        # If not connected, try to reconnect
        if self.state != ServerState.CONNECTED:
            reconnected = await self.initialize()
            if not reconnected:
                raise RuntimeError(f"Cannot get prompt: server {self.name} is not connected")
        
        # Use circuit breaker pattern to prevent cascading failures
        try:
            return await self.circuit_breaker.execute(
                lambda: self.retry_handler.execute(self._get_prompt_internal, name, arguments or {})
            )
        except Exception as e:
            logging.error(f"Failed to get prompt {name} after retries: {e}")
            self.state = ServerState.FAILED
            await self.cleanup()
            self.last_connection_attempt = time.time()
            raise RuntimeError(f"Prompt execution failed: {str(e)}")

    async def _get_prompt_internal(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Internal prompt execution function wrapped by retry handler.
        
        Args:
            name: The name of the prompt.
            arguments: Arguments for the prompt.
            
        Returns:
            Prompt result with messages.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        
        logging.info(f"Getting prompt {name}...")
        result = await self.session.get_prompt(name, arguments)
        return result
    
    async def initialize(self) -> bool:
        """Initialize the server connection with reconnection logic and timeout.
        
        Returns:
            bool: True if connection succeeded, False otherwise
        """
        if self.state == ServerState.CONNECTING:
            logging.info(f"Server {self.name} already connecting, waiting...")
            return False
                
        self.state = ServerState.CONNECTING
        self.last_connection_attempt = time.time()
        
        server_params = StdioServerParameters(
            command=shutil.which("npx") if self.config['command'] == "npx" else self.config['command'],
            args=self.config['args'],
            env={**os.environ, **self.config['env']} if self.config.get('env') else None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        try:
            # Use circuit breaker pattern for initialization with timeout
            init_task = asyncio.create_task(
                self.circuit_breaker.execute(lambda: self._initialize_internal(server_params))
            )
            
            # Add timeout
            await asyncio.wait_for(init_task, timeout=self.global_config.server_init_timeout)
            
            # Connection successful, reset failure counter
            self.connection_failures = 0
            self.state = ServerState.CONNECTED
            
            # Start health check task
            self._start_health_check()
            
            return True
                
        except asyncio.TimeoutError:
            logging.error(f"Timeout initializing server {self.name} after {self.global_config.server_init_timeout} seconds")
            self.connection_failures += 1
            self.state = ServerState.FAILED
            await self.cleanup()
            return False
        except Exception as e:
            self.connection_failures += 1
            logging.error(f"Error initializing server {self.name}: {e}")
            self.state = ServerState.FAILED
            await self.cleanup()
            return False

    async def _initialize_internal(self, server_params: StdioServerParameters) -> None:
        """Internal initialization function wrapped by circuit breaker.
        
        Args:
            server_params: Server parameters for initialization
        """
        logging.info(f"Connecting to server {self.name}...")
        self.stdio_context = stdio_client(server_params)
        read, write = await self.stdio_context.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        self.capabilities = await self.session.initialize()
        logging.info(f"Server {self.name} connected successfully")
        
        # Fetch tools immediately after successful connection
        await self._update_tools()

    async def _update_tools(self) -> None:
        """Update the list of available tools, resources, and prompts from the server."""
        if not self.session:
            return
            
        try:
            # Fetch and cache tools
            tools_response = await self.session.list_tools()
            self.tools = []
            
            supports_progress = (
                self.capabilities 
                and 'progress' in self.capabilities
            )
            
            if supports_progress:
                logging.info(f"Server {self.name} supports progress tracking")
            
            for item in tools_response:
                if isinstance(item, tuple) and item[0] == 'tools':
                    for tool in item[1]:
                        self.tools.append(Tool(tool.name, tool.description, tool.inputSchema))
                        if supports_progress:
                            logging.info(f"Tool '{tool.name}' will support progress tracking")
            
            # Check if the server supports resources capability and fetch them
            if self.capabilities and 'resources' in self.capabilities:
                logging.info(f"Server {self.name} supports resources")
                try:
                    resources_response = await self.session.list_resources()
                    self.resources = []
                    if hasattr(resources_response, 'resources'):
                        self.resources = resources_response.resources
                    elif isinstance(resources_response, tuple) and resources_response[0] == 'resources':
                        self.resources = resources_response[1]
                    logging.info(f"Cached {len(self.resources)} resources from server {self.name}")
                except Exception as e:
                    logging.error(f"Error fetching resources from server {self.name}: {e}")
            
            # Check if the server supports prompts capability and fetch them
            if self.capabilities and 'prompts' in self.capabilities:
                logging.info(f"Server {self.name} supports prompts")
                try:
                    prompts_response = await self.session.list_prompts()
                    self.prompts = []
                    if hasattr(prompts_response, 'prompts'):
                        self.prompts = prompts_response.prompts
                    logging.info(f"Cached {len(self.prompts)} prompts from server {self.name}")
                except Exception as e:
                    logging.error(f"Error fetching prompts from server {self.name}: {e}")
                
        except Exception as e:
            logging.error(f"Error updating tools for server {self.name}: {e}")

    def _start_health_check(self) -> None:
        """Start periodic health check task."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
    async def _health_check_loop(self) -> None:
        """Periodic health check to ensure server connection is still alive."""
        while True:
            try:
                await asyncio.sleep(self.global_config.health_check_interval)
                if self.state == ServerState.CONNECTED:
                    # Ping the server by listing tools
                    await self._update_tools()
                    logging.debug(f"Health check passed for server {self.name}")
                elif self.state in [ServerState.DISCONNECTED, ServerState.FAILED]:
                    # Try to reconnect if disconnected
                    if time.time() - self.last_connection_attempt > self.reconnect_delay:
                        logging.info(f"Attempting to reconnect to server {self.name}...")
                        await self.initialize()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.warning(f"Health check failed for server {self.name}: {e}")
                if self.state == ServerState.CONNECTED:
                    self.state = ServerState.FAILED
                    await self.cleanup()
                    # Schedule reconnection
                    self.last_connection_attempt = time.time()

    async def list_tools(self) -> List[Any]:
        """List available tools from the server with reconnection logic.
        
        Returns:
            A list of available tools.
        """
        # If disconnected, try to reconnect
        if self.state != ServerState.CONNECTED:
            if time.time() - self.last_connection_attempt > self.reconnect_delay:
                reconnected = await self.initialize()
                if not reconnected:
                    return []
            else:
                return []
        
        # Return cached tools list to avoid frequent server calls
        return self.tools

    async def execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Any:
        """Execute a tool with advanced retry and reconnection logic.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            
        Returns:
            Tool execution result.
            
        Raises:
            RuntimeError: If server is not initialized or tool execution fails.
        """
        # If not connected, try to reconnect
        if self.state != ServerState.CONNECTED:
            reconnected = await self.initialize()
            if not reconnected:
                raise RuntimeError(f"Cannot execute tool: server {self.name} is not connected")
        
        # Use circuit breaker pattern to prevent cascading failures
        try:
            return await self.circuit_breaker.execute(
                lambda: self.retry_handler.execute(self._execute_tool_internal, tool_name, arguments)
            )
        except Exception as e:
            logging.error(f"Failed to execute tool {tool_name} after retries: {e}")
            # Mark server as failed if execution consistently fails
            self.state = ServerState.FAILED
            await self.cleanup()
            self.last_connection_attempt = time.time()
            raise RuntimeError(f"Tool execution failed: {str(e)}")

    async def _execute_tool_internal(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Internal tool execution function wrapped by retry handler.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            
        Returns:
            Tool execution result.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        supports_progress = (
            self.capabilities 
            and 'progress' in self.capabilities
        )

        if supports_progress:
            logging.info(f"Executing {tool_name} with progress tracking...")
            result = await self.session.call_tool(
                tool_name, 
                arguments,
                progress_token=f"{tool_name}_execution"
            )
        else:
            logging.info(f"Executing {tool_name}...")
            result = await self.session.call_tool(tool_name, arguments)

        return result

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            # Cancel health check task
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                self._health_check_task = None
                
            try:
                if self.session:
                    try:
                        await self.session.__aexit__(None, None, None)
                    except Exception as e:
                        logging.warning(f"Warning during session cleanup for {self.name}: {e}")
                    finally:
                        self.session = None

                if self.stdio_context:
                    try:
                        await self.stdio_context.__aexit__(None, None, None)
                    except (RuntimeError, asyncio.CancelledError) as e:
                        logging.info(f"Note: Normal shutdown message for {self.name}: {e}")
                    except Exception as e:
                        logging.warning(f"Warning during stdio cleanup for {self.name}: {e}")
                    finally:
                        self.stdio_context = None
                        
                if self.state != ServerState.DISCONNECTED:
                    self.state = ServerState.DISCONNECTED
                    
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.
        
        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if 'properties' in self.input_schema:
            for param_name, param_info in self.input_schema['properties'].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get('required', []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)
        
        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


# Provider registry for centralized provider management
class ProviderRegistry:
    """Manages the registration and access of LLM providers."""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.providers = {}
        self._register_default_providers()
        
    def _register_default_providers(self):
        """Register the default LLM providers."""
        self.register_provider(
            "groq", 
            "https://api.groq.com/openai/v1/chat/completions",
            "https://api.groq.com/openai/v1/models",
            lambda api_key: {"Authorization": f"Bearer {api_key}"},
            lambda resp: [model["id"] for model in resp.get("data", [])],
            ["llama-3.2-90b-vision-preview", "llama-3.2-70b-instruct-preview"]
        )
        
        self.register_provider(
            "openai",
            "https://api.openai.com/v1/chat/completions",
            "https://api.openai.com/v1/models",
            lambda api_key: {"Authorization": f"Bearer {api_key}"},
            lambda resp: [m["id"] for m in resp.get("data", []) if "gpt" in m["id"].lower()],
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        )
        
        self.register_provider(
            "anthropic",
            "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com/v1/models",
            lambda api_key: {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            lambda resp: [model["id"] for model in resp.get("data", [])],
            ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"]
        )
        
        self.register_provider(
            "gemini",
            None,  # No direct URL for the Google SDK
            None,  # No models URL for the Google SDK
            lambda api_key: {"Authorization": f"Bearer {api_key}"},
            lambda resp: [model["id"] for model in resp],
            ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
        )
        
        self.register_provider(
            "openroute",
            "https://openrouter.ai/api/v1/chat/completions",
            "https://openrouter.ai/api/v1/models",
            lambda api_key: {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost",
                "X-Title": "MCP-Chatbot"
            },
            lambda resp: [model["id"] for model in resp],
            ["openai/gpt-4o", "anthropic/claude-3-5-sonnet", "meta-llama/llama-3.2-70b-instruct"]
        )
        
        self.register_provider(
            "ollama",
            None,  # Set dynamically based on host
            None,  # Set dynamically based on host
            lambda api_key: {},  # No auth for Ollama
            lambda resp: [model["name"] for model in resp],
            ["llama3", "mistral", "mixtral"]
        )
            
    def register_provider(self, name, url, models_url, headers_func, models_extractor, default_models):
        """Register a new provider."""
        self.providers[name] = {
            "url": url,
            "models_url": models_url,
            "headers": headers_func,
            "models_extractor": models_extractor,
            "default_models": default_models
        }
    
    def get_provider(self, name):
        """Get a provider by name."""
        return self.providers.get(name.lower())
    
    def get_all_providers(self):
        """Get all registered providers."""
        return self.providers


class LLMClient:
    """Manages communication with multiple LLM providers using async HTTP."""
    
    def __init__(
        self, 
        config: Configuration,
        provider: str = None,
        model: str = None
    ) -> None:
        self.config = config
        self.provider = provider or config.default_provider
        self.model = model or config.default_model
        self.api_key = config.get_api_key(self.provider) if self.provider != "ollama" else None
        
        # Initialize provider registry
        self.provider_registry = ProviderRegistry(config)
        self.PROVIDER_CONFIGS = self.provider_registry.providers
        
        # Store available models for each provider
        self.available_models = {}
        
        # Create session for HTTP requests
        self.session = None
        self.session_lock = asyncio.Lock()
        
        # Create retry handler
        self.retry_handler = AsyncRetry(
            max_retries=config.max_retries,
            base_delay=config.retry_delay_base,
            max_delay=config.retry_max_delay
        )
        
        # Create circuit breakers for each provider
        self.circuit_breakers = {
            provider: CircuitBreaker() 
            for provider in self.PROVIDER_CONFIGS.keys()
        }
        
        # Health check state
        self.provider_health = {provider: True for provider in self.PROVIDER_CONFIGS.keys()}
        self._health_check_task = None
        self.last_token_usage = TokenUsage()
        self.using_default_models = {provider: False for provider in self.provider_registry.providers.keys()}

    # Add this method to LLMClient
    def estimate_tokens(self, text: str, model: str = None) -> int:
        """Estimate token count using appropriate tokenizer."""
        model = model or self.model
        
        # Use tiktoken for OpenAI/compatible models if available
        if TIKTOKEN_AVAILABLE and self.provider in ["openai", "groq"]:
            try:
                if "gpt-4" in model:
                    encoding = tiktoken.encoding_for_model("gpt-4")
                elif "gpt-3.5" in model:
                    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                else:
                    encoding = tiktoken.get_encoding("cl100k_base")  # Default for newer models
                return len(encoding.encode(text))
            except Exception as e:
                logging.warning(f"Error using tiktoken: {e}")
        
        # Fallback approximation
        return len(text.split()) * 1.3
                
    async def _test_provider_connection(self, provider: str) -> bool:
        """Test actual API connection without fallbacks.
        
        Args:
            provider: The provider to test
            
        Returns:
            True if connection succeeds, raises exception otherwise
        """
        if provider not in self.PROVIDER_CONFIGS:
            raise ValueError(f"Unsupported provider: {provider}")
            
        config = self.PROVIDER_CONFIGS[provider]
        api_key = None if provider == "ollama" else self.config.get_api_key(provider)
        
        if provider == "ollama":
            url = f"{self.config.ollama_host.rstrip('/')}/api/tags"
            session = await self._ensure_session()
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                return True
        elif config["models_url"]:
            headers = config["headers"](api_key)
            session = await self._ensure_session()
            async with session.get(config["models_url"], headers=headers, timeout=10) as response:
                response.raise_for_status()
                return True
        else:
            # For providers without direct URL access (like Gemini)
            if provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                await asyncio.to_thread(genai.list_models)
                return True
                
        return True  # If we get here, connection succeeded

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists and create if needed.
        
        Returns:
            aiohttp.ClientSession: The HTTP session
        """
        async with self.session_lock:
            if self.session is None or self.session.closed:
                # Configure timeout and connection limits
                timeout = aiohttp.ClientTimeout(
                    connect=self.config.connection_timeout,
                    total=self.config.read_timeout
                )
                connector = aiohttp.TCPConnector(
                    limit=20,  # Connection pool size
                    limit_per_host=8,  # Connections per host
                    ttl_dns_cache=300,  # DNS cache TTL
                    enable_cleanup_closed=True,
                    force_close=False,  # Allow connection reuse
                    keepalive_timeout=60  # Keep connections alive
                )
                self.session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                    raise_for_status=True,
                    # Add common headers for all requests
                    headers={
                        "User-Agent": "MCP-Client/1.0",
                    }
                )
                
        return self.session

    async def initialize(self) -> None:
        """Initialize the client and fetch initial model information."""
        await self._ensure_session()
        
        # Start health checks
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Fetch models for all providers in parallel
        fetch_tasks = []
        for provider_name in self.PROVIDER_CONFIGS.keys():
            # Only try to fetch if we have an API key (except for Ollama)
            if provider_name == "ollama" or self.config.get_api_key(provider_name) is not None:
                task = asyncio.create_task(self._fetch_provider_models(provider_name))
                fetch_tasks.append(task)
        
        # Wait for all fetches to complete (don't raise exceptions)
        if fetch_tasks:
            await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
        # Ensure we have models for the current provider
        if self.provider not in self.available_models:
            self.available_models[self.provider] = self.PROVIDER_CONFIGS[self.provider]["default_models"]
            
    async def _fetch_provider_models(self, provider: str) -> None:
        """Fetch models for a specific provider with error handling.
        
        Args:
            provider: The provider to fetch models for
        """
        try:
            await self._fetch_available_models(provider)
            logging.info(f"Successfully loaded {len(self.available_models[provider])} models for {provider}")
        except Exception as e:
            logging.warning(f"Could not fetch models for {provider}: {e}")
            # Use default models as fallback
            self.available_models[provider] = self.PROVIDER_CONFIGS[provider]["default_models"]
    
    async def _health_check_loop(self) -> None:
        """Periodically check the health of LLM providers."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Create tasks to check all providers in parallel
                health_check_tasks = []
                
                # Check current provider first with higher priority
                check_task = asyncio.create_task(
                    self._check_provider_health(self.provider)
                )
                health_check_tasks.append(check_task)
                
                # Check other providers in the background
                for provider in self.PROVIDER_CONFIGS.keys():
                    if provider != self.provider:
                        # Only check if we have an API key
                        if provider == "ollama" or self.config.get_api_key(provider):
                            task = asyncio.create_task(
                                self._check_provider_health(provider)
                            )
                            health_check_tasks.append(task)
                
                # Wait for all health checks to complete
                await asyncio.gather(*health_check_tasks, return_exceptions=True)
                
                # Log overall health status
                healthy_providers = [p for p, status in self.provider_health.items() if status]
                logging.debug(f"Provider health status: {len(healthy_providers)}/{len(self.provider_health)} healthy")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying on error
    
    async def _check_provider_health(self, provider: str) -> None:
        """Check health of a specific provider.
        
        Args:
            provider: The provider to check
        """
        # Store the original models to detect if we're using defaults
        original_models = self.available_models.get(provider, [])
        default_models = self.PROVIDER_CONFIGS[provider]["default_models"]
        
        try:
            # Try to actually connect to the API
            api_key = None if provider == "ollama" else self.config.get_api_key(provider)
            if not api_key and provider != "ollama":
                self.provider_health[provider] = False
                return
                
            # Test actual API connection - use a separate method to test connection
            await self._test_provider_connection(provider)
            
            # If connection succeeds, update health status
            was_unhealthy = not self.provider_health.get(provider, True)
            self.provider_health[provider] = True
            
            if was_unhealthy:
                logging.info(f"Provider {provider} has recovered and is now healthy")
                
        except Exception as e:
            # Provider is unhealthy
            was_healthy = self.provider_health.get(provider, True)
            self.provider_health[provider] = False
            
            if was_healthy:
                logging.warning(f"Health check failed for provider {provider}: {e}")
            else:
                logging.debug(f"Provider {provider} still unhealthy: {e}")
                
    async def _fetch_available_models(self, provider: str) -> List[str]:
        """Fetch available models from the provider's API asynchronously.
        
        Args:
            provider: The provider to fetch models for
            
        Returns:
            A list of available model IDs
            
        Raises:
            ValueError: If the provider is not supported
            aiohttp.ClientError: If the API request fails
        """
        if provider not in self.PROVIDER_CONFIGS:
            raise ValueError(f"Unsupported provider: {provider}")
        self.provider_health[provider] = False
        # Get provider configuration
        config = self.PROVIDER_CONFIGS[provider]
        
        # Provider-specific implementations
        if provider == "groq":
            return await self._fetch_groq_models(config)
        elif provider == "openai":
            return await self._fetch_openai_models(config)
        elif provider == "gemini":
            return await self._fetch_gemini_models(config)
        elif provider == "anthropic":
            return await self._fetch_anthropic_models(config)
        elif provider == "openroute":
            return await self._fetch_openroute_models(config)
        elif provider == "ollama":
            return await self._fetch_ollama_models(config)
        else:
            # Generic fallback for any other providers
            return await self._fetch_generic_models(provider, config)
    
    async def _fetch_groq_models(self, config: Dict[str, Any]) -> List[str]:
        """Fetch Groq models with specialized handling."""
        try:
            api_key = self.config.get_api_key("groq")
            if not api_key:
                self.provider_health["groq"] = False
                return []
                    
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            session = await self._ensure_session()
            async with session.get(config["models_url"], headers=headers) as response:
                data = await response.json()
                logging.debug(f"Groq API response: {data}")
                    
                if "data" in data and isinstance(data["data"], list):
                    models = [model["id"] for model in data["data"] if "id" in model]
                    
                    if models:
                        self.available_models["groq"] = models
                        self.provider_health["groq"] = True
                        return models
                    
                logging.warning("Could not extract models from Groq API response")
                self.provider_health["groq"] = False
                return []
                    
        except Exception as e:
            logging.warning(f"Error fetching Groq models: {e}")
            self.provider_health["groq"] = False
            return []

    async def _fetch_openai_models(self, config: Dict[str, Any]) -> List[str]:
        """Fetch OpenAI models with specialized handling."""
        try:
            api_key = self.config.get_api_key("openai")
            if not api_key:
                self.provider_health["openai"] = False
                return []
                    
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            session = await self._ensure_session()
            async with session.get(config["models_url"], headers=headers) as response:
                data = await response.json()
                logging.debug(f"OpenAI API response: {data}")
                    
                if "data" in data and isinstance(data["data"], list):
                    # Filter for only GPT models
                    models = [m["id"] for m in data["data"] if "id" in m and "gpt" in m["id"].lower()]
                    
                    if models:
                        self.available_models["openai"] = models
                        self.provider_health["openai"] = True
                        return models
                    
                logging.warning("Could not extract models from OpenAI API response")
                self.provider_health["openai"] = False
                return []
                    
        except Exception as e:
            logging.warning(f"Error fetching OpenAI models: {e}")
            self.provider_health["openai"] = False
            return []
        
    async def _fetch_gemini_models(self, config: Dict[str, Any]) -> List[str]:
        """Fetch Gemini models using the official Google SDK."""
        try:
            api_key = self.config.get_api_key("gemini")
            if not api_key:
                self.provider_health["gemini"] = False
                return []
                    
            # Using the official Google SDK to get models
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                    
                # Use asyncio.to_thread to run the blocking SDK call in a separate thread
                models = await asyncio.to_thread(genai.list_models)
                    
                if models:
                    # Filter for only Gemini models
                    gemini_models = [model.name for model in models if "gemini" in model.name.lower()]
                    
                    if gemini_models:
                        self.available_models["gemini"] = gemini_models
                        self.provider_health["gemini"] = True
                        return gemini_models
            except ImportError:
                logging.warning("Google GenerativeAI SDK not installed.")
            except Exception as sdk_err:
                logging.warning(f"Error using Google SDK: {sdk_err}")
            
            # If we get here, something failed
            self.provider_health["gemini"] = False
            return []
                    
        except Exception as e:
            logging.warning(f"Error fetching Gemini models: {e}")
            self.provider_health["gemini"] = False
            return []

    async def _fetch_anthropic_models(self, config: Dict[str, Any]) -> List[str]:
        """Fetch Anthropic models with specialized handling."""
        try:
            api_key = self.config.get_api_key("anthropic")
            if not api_key:
                self.provider_health["anthropic"] = False
                return []
                    
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            
            session = await self._ensure_session()
            async with session.get(config["models_url"], headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                data = await response.json()
                logging.debug(f"Anthropic API response: {data}")
                    
                if "data" in data and isinstance(data["data"], list):
                    models = [model["id"] for model in data["data"] if "id" in model]
                    
                    if models:
                        self.available_models["anthropic"] = models
                        self.provider_health["anthropic"] = True
                        return models
                    
                logging.warning("Could not extract models from Anthropic API response")
                self.provider_health["anthropic"] = False
                return []
                    
        except Exception as e:
            logging.warning(f"Error fetching Anthropic models: {e}")
            self.provider_health["anthropic"] = False
            return []

    async def _fetch_openroute_models(self, config: Dict[str, Any]) -> List[str]:
        """Fetch OpenRoute models with specialized handling."""
        try:
            api_key = self.config.get_api_key("openroute")
            if not api_key:
                self.provider_health["openroute"] = False
                return []
                    
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost",
                "X-Title": "MCP-Chatbot"
            }
            
            session = await self._ensure_session()
            async with session.get(config["models_url"], headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                data = await response.json()
                logging.debug(f"OpenRoute API response type: {type(data)}")
                logging.debug(f"OpenRoute API response: {data}")
                    
                models = []
                # Handle different possible response formats
                if isinstance(data, list):
                    # Direct array of models
                    for model in data:
                        if isinstance(model, dict) and "id" in model:
                            models.append(model["id"])
                elif isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list):
                        # Models in a "data" key
                        for model in data["data"]:
                            if isinstance(model, dict) and "id" in model:
                                models.append(model["id"])
                    elif "models" in data and isinstance(data["models"], list):
                        # Models in a "models" key
                        for model in data["models"]:
                            if isinstance(model, dict) and "id" in model:
                                models.append(model["id"])
                    
                if models:
                    self.available_models["openroute"] = models
                    self.provider_health["openroute"] = True
                    return models
                    
                logging.warning("Could not extract models from OpenRoute API response")
                self.provider_health["openroute"] = False
                return []
        except Exception as e:
            logging.warning(f"Error fetching OpenRoute models: {e}")
            self.provider_health["openroute"] = False
            return []

    async def _fetch_ollama_models(self, config: Dict[str, Any]) -> List[str]:
        """Fetch Ollama models from API with improved error handling and response processing."""
        # Get and normalize Ollama host URL
        ollama_host = self.config.ollama_host or "http://localhost:11434"
        ollama_host = ollama_host.rstrip('/')
        logging.debug(f"Base Ollama host URL: {ollama_host}")
            
        # Define potential API endpoints to try
        standard_endpoint = f"{ollama_host}/api/tags"
        alternate_endpoint = f"{ollama_host}/v1/api/tags"
            
        # Get session with proper error handling
        try:
            session = await self._ensure_session()
        except Exception as e:
            logging.error(f"Failed to create HTTP session: {e}")
            self.provider_health["ollama"] = False
            return []
            
        # Define timeout
        timeout = aiohttp.ClientTimeout(total=15)
            
        # Try standard endpoint first
        logging.debug(f"Attempting to fetch models from standard endpoint: {standard_endpoint}")
        try:
            models = await self._try_fetch_from_endpoint(session, standard_endpoint, timeout, [])
            if models:
                self.provider_health["ollama"] = True
                return models
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                logging.info(f"Standard endpoint returned 404, trying alternate endpoint with /v1/ prefix")
                try:
                    models = await self._try_fetch_from_endpoint(session, alternate_endpoint, timeout, [])
                    if models:
                        self.provider_health["ollama"] = True
                        return models
                except Exception as e2:
                    logging.error(f"Failed to fetch models from alternate endpoint: {e2}")
            else:
                logging.error(f"HTTP error with Ollama API: {e.status}, message='{e.message}'")
        except Exception as e:
            logging.error(f"Error fetching models from standard endpoint: {e}")
            
        # If all endpoints failed, mark as unhealthy
        logging.warning("All Ollama API endpoints failed")
        self.provider_health["ollama"] = False
        return []

    async def _try_fetch_from_endpoint(self, session, endpoint, timeout, default_models):
        """Helper method to try fetching models from a specific endpoint."""
        async with session.get(endpoint, timeout=timeout) as response:
            response.raise_for_status()
                
            # Parse JSON response
            data = await response.json()
            logging.debug(f"Ollama API response from {endpoint}: {data}")
                
            # Extract model names from the response
            models = []
                
            if isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
                # Process API format
                for model in data["models"]:
                    if isinstance(model, dict) and "name" in model:
                        models.append(model["name"])
                
            if models:
                logging.info(f"Successfully fetched {len(models)} models from {endpoint}")
                self.available_models["ollama"] = models
                return models
            else:
                logging.warning(f"No models found in response from {endpoint}")
                return []

    async def _fetch_generic_models(self, provider: str, config: Dict[str, Any]) -> List[str]:
        """Generic model fetching for other providers."""
        try:
            api_key = self.config.get_api_key(provider)
            if not api_key:
                self.provider_health[provider] = False
                return []
                    
            # Prepare headers using the config's header function
            headers = config["headers"](api_key)
                
            session = await self._ensure_session()
            async with session.get(config["models_url"], headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                data = await response.json()
                logging.debug(f"{provider.capitalize()} API response: {data}")
                    
                # Use the provider's extractor function to get models
                try:
                    models = config["models_extractor"](data)
                        
                    if models:
                        self.available_models[provider] = models
                        self.provider_health[provider] = True
                        return models
                except Exception as e:
                    logging.warning(f"Error extracting models for {provider}: {e}")
                    
                # If extraction failed, mark as unhealthy
                self.provider_health[provider] = False
                return []
        except Exception as e:
            logging.warning(f"Error fetching models for {provider}: {e}")
            self.provider_health[provider] = False
            return []

    async def get_response(self, messages: List[Dict[str, str]]) -> Tuple[str, TokenUsage]:
        """Get a response from the LLM asynchronously.
        
        Args:
            messages: A list of message dictionaries.
            
        Returns:
            The LLM's response as a string.
        """
        # Check if we need to reconnect to a different provider
        # if not self.provider_health[self.provider]:
        #     logging.warning(f"Current provider {self.provider} is unhealthy")
            
        #     # Find a healthy alternative provider
        #     for alt_provider, is_healthy in self.provider_health.items():
        #         if is_healthy and (alt_provider == "ollama" or self.config.get_api_key(alt_provider)):
        #             logging.info(f"Switching to healthy provider: {alt_provider}")
        #             await self.change_provider(alt_provider)
        #             break
        if not self.provider_health[self.provider]:
            logging.warning(f"Current provider {self.provider} is unhealthy")
            # Just log the warning without auto-switching
            logging.info("Use '/switch <provider> <model>' to manually change providers")
            
        # Use retry mechanism with circuit breaker for resiliency
        try:
            circuit_breaker = self.circuit_breakers[self.provider]
            
            # return await circuit_breaker.execute(
            #     lambda: self.retry_handler.execute(self._get_provider_response, messages)
            # )
            response_text = await circuit_breaker.execute(
                lambda: self.retry_handler.execute(self._get_provider_response, messages)
            )
            return response_text, self.last_token_usage
        except Exception as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            
            # Mark provider as unhealthy
            self.provider_health[self.provider] = False
            
            # Try to find a fallback provider
            # for alt_provider, is_healthy in self.provider_health.items():
            #     if is_healthy and alt_provider != self.provider and (
            #         alt_provider == "ollama" or self.config.get_api_key(alt_provider)
            #     ):
            #         logging.info(f"Falling back to alternative provider: {alt_provider}")
            #         try:
            #             await self.change_provider(alt_provider)
            #             return await self.retry_handler.execute(self._get_provider_response, messages)
            #         except Exception as fallback_error:
            #             logging.error(f"Fallback to {alt_provider} also failed: {fallback_error}")
            #             break
            # Just log the error without auto-switching
            logging.error("Connection to current provider failed. Use '/switch <provider> <model>' to change providers")
            error_response = f"I encountered an error connecting to the language model service. Please try again in a moment. (Error: {error_message})"
            return error_response, TokenUsage()  # Return empty token usage on error

    async def _get_provider_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the current provider.
        
        Args:
            messages: A list of message dictionaries.
            
        Returns:
            The LLM's response as a string.
        """
        try:
            if self.provider.lower() == "groq":
                return await self._call_groq(messages)
            elif self.provider.lower() == "openai":
                return await self._call_openai(messages)
            elif self.provider.lower() == "anthropic":
                return await self._call_anthropic(messages)
            elif self.provider.lower() == "gemini":
                return await self._call_gemini(messages)
            elif self.provider.lower() == "openroute":
                return await self._call_openroute(messages)
            elif self.provider.lower() == "ollama":
                return await self._call_ollama(messages)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except aiohttp.ClientError as e:
            error_message = f"HTTP error: {str(e)}"
            logging.error(error_message)
            
            # Add more detailed error information if available
            if hasattr(e, 'status'):
                status_code = e.status
                logging.error(f"Status code: {status_code}")
            
            if hasattr(e, 'message'):
                logging.error(f"Response details: {e.message}")
            
            raise
        except asyncio.TimeoutError:
            logging.error(f"Request to {self.provider} timed out")
            raise RuntimeError(f"Request to {self.provider} timed out after {self.config.read_timeout} seconds")
        except Exception as e:
            logging.error(f"Error calling {self.provider}: {str(e)}")
            raise
    
    async def change_provider(self, provider: str, model: str = None) -> None:
        """Change the LLM provider and model asynchronously."""
        # Check if provider is supported
        if provider.lower() not in self.PROVIDER_CONFIGS:
            supported = ", ".join(self.PROVIDER_CONFIGS.keys())
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {supported}")
        
        # Set the provider
        provider = provider.lower()
        
        # Fetch available models for this provider if needed
        if provider not in self.available_models or not self.available_models[provider]:
            try:
                await self._fetch_available_models(provider)
            except Exception as e:
                logging.warning(f"Could not fetch models for {provider}: {e}")
                # No fallback to default models
                self.available_models[provider] = []
        
        # Make sure there are models available
        if not self.available_models[provider]:
            raise ValueError(f"No models available for {provider}. API connection may have failed.")
        
        # Update the provider
        self.provider = provider
        
        # Check if model is supported and set it
        if model:
            if model not in self.available_models[provider]:
                raise ValueError(f"Model '{model}' is not available for {provider}. Use /llm to see available models.")
            self.model = model
        else:
            # Use the first available model
            self.model = self.available_models[provider][0]
        
        # Set the API key
        if provider != "ollama":
            self.api_key = self.config.get_api_key(provider)
        else:
            self.api_key = None
                
        logging.info(f"Switched to {provider} provider with model {self.model}")

    async def _call_groq(self, messages: List[Dict[str, str]]) -> str:
        """Call the Groq API asynchronously."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": messages,
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None
        }
        
        session = await self._ensure_session()
        async with session.post(url, headers=headers, json=payload) as response:
            data = await response.json()
            # Extract token usage (Groq API provides similar structure to OpenAI)
            if "usage" in data:
                self.last_token_usage.update(
                    prompt_tokens=data["usage"].get("prompt_tokens", 0),
                    completion_tokens=data["usage"].get("completion_tokens", 0),
                    total_tokens=data["usage"].get("total_tokens", 0)
                )
            return data['choices'][0]['message']['content']

    async def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        """Call the OpenAI API asynchronously."""
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": messages,
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False
        }
        
        session = await self._ensure_session()
        # async with session.post(url, headers=headers, json=payload) as response:
        #     data = await response.json()
        #     return data['choices'][0]['message']['content']
        async with session.post(url, headers=headers, json=payload) as response:
            data = await response.json()
            # Extract token usage
            if "usage" in data:
                self.last_token_usage.update(
                    prompt_tokens=data["usage"].get("prompt_tokens", 0),
                    completion_tokens=data["usage"].get("completion_tokens", 0),
                    total_tokens=data["usage"].get("total_tokens", 0)
                )
            return data['choices'][0]['message']['content']

    async def _call_anthropic(self, messages: List[Dict[str, str]]) -> str:
        """Call the Anthropic API asynchronously."""
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Convert OpenAI message format to Anthropic format
        anthropic_messages = []
        system_content = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        payload = {
            "messages": anthropic_messages,
            "model": self.model,
            "max_tokens": 4096,
            "temperature": 0.7
        }
        
        if system_content:
            payload["system"] = system_content
        
        session = await self._ensure_session()
        async with session.post(url, headers=headers, json=payload) as response:
            data = await response.json()
            
            # Extract token usage from Anthropic response
            if "usage" in data:
                self.last_token_usage.update(
                    prompt_tokens=data["usage"].get("input_tokens", 0),
                    completion_tokens=data["usage"].get("output_tokens", 0)
                )
            
            # Existing content extraction...
            if isinstance(data['content'], list) and len(data['content']) > 0:
                return data['content'][0]['text']
            else:
                return "No text content returned from Anthropic API"

    async def _call_gemini(self, messages: List[Dict[str, str]]) -> str:
        """Call the Google Gemini API using the official method."""
        # First ensure we have the required package
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("The 'google-generativeai' package is required. Install it with 'pip install google-generativeai'")
        
        # Configure the API with the provided key
        genai.configure(api_key=self.api_key)
        
        prompt_text = "\n".join([msg.get("content", "") for msg in messages])
        
        # Convert OpenAI message format to Gemini format
        system_content = None
        conversation = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                conversation.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                conversation.append({"role": "model", "parts": [{"text": msg["content"]}]})
        
        # Initialize the model with generation config
        generation_config = genai.GenerationConfig(
            temperature=0.7,
            top_p=1,
            top_k=1,
            max_output_tokens=4096,
        )
        
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        
        try:
            # Check if we have multiple messages (conversation)
            if len(conversation) > 1:
                # Start a chat session WITHOUT system instruction
                chat = model.start_chat(history=[])
                
                # If we have system content, add it as a separate message
                if system_content:
                    # Some implementations use a special format for system instructions
                    # We'll add it as a regular message with a clear prefix
                    chat.history.append({
                        "role": "user", 
                        "parts": [{"text": f"[SYSTEM INSTRUCTION]: {system_content}"}]
                    })
                    chat.history.append({
                        "role": "model",
                        "parts": [{"text": "I'll follow these instructions."}]
                    })
                
                # Add previous messages to the chat
                for message in conversation[:-1]:
                    chat.history.append(message)
                
                # Send the last message to get a response
                if conversation:
                    last_message = conversation[-1]["parts"][0]["text"]
                    response = await asyncio.to_thread(chat.send_message, last_message)
                else:
                    # Should never happen but just in case
                    response = await asyncio.to_thread(chat.send_message, "Hello")
            else:
                # Simple single message case
                prompt = ""
                if conversation:
                    prompt = conversation[0]["parts"][0]["text"]
                
                # For single messages, use generate_content and pass system instruction directly
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    # Some Gemini versions support system instructions directly in generate_content
                    system_instruction=system_content if system_content else None
                )
        except TypeError as e:
            # If there's an issue with system_instruction parameter
            if "system_instruction" in str(e):
                logging.warning("This version of Gemini API doesn't support system_instruction parameter")
                if len(conversation) > 1:
                    chat = model.start_chat()
                    
                    # Add system content as a user message
                    if system_content:
                        await asyncio.to_thread(
                            chat.send_message,
                            f"[SYSTEM INSTRUCTION]: {system_content}\nPlease acknowledge and follow these instructions."
                        )
                    
                    # Add all messages except the last one
                    for i, message in enumerate(conversation[:-1]):
                        if message["role"] == "user":
                            await asyncio.to_thread(
                                chat.send_message,
                                message["parts"][0]["text"]
                            )
                    
                    # Send the last message
                    if conversation:
                        last_message = conversation[-1]["parts"][0]["text"]
                        response = await asyncio.to_thread(chat.send_message, last_message)
                    else:
                        response = await asyncio.to_thread(chat.send_message, "Hello")
                else:
                    # Simple case with just one message
                    prompt = conversation[0]["parts"][0]["text"] if conversation else ""
                    if system_content:
                        prompt = f"[SYSTEM INSTRUCTION]: {system_content}\n\n{prompt}"
                    
                    response = await asyncio.to_thread(model.generate_content, prompt)
            else:
                raise
        # Extract and return the response text
        response_text = ""
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'parts') and len(response.parts) > 0:
            response_text = response.parts[0].text
        else:
            response_text = str(response)
        
        # Try to extract token counts from response metadata
        prompt_tokens = 0
        completion_tokens = 0
        
        # Look for token counts in response object
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            if hasattr(usage, 'prompt_token_count'):
                prompt_tokens = usage.prompt_token_count
            if hasattr(usage, 'candidates_token_count'):
                completion_tokens = usage.candidates_token_count
        
        # If we couldn't find token info, estimate tokens
        if prompt_tokens == 0:
            prompt_tokens = self.estimate_tokens(prompt_text)
        if completion_tokens == 0:
            completion_tokens = self.estimate_tokens(response_text)
        
        # Update token usage tracker
        self.last_token_usage.update(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        
        return response_text

    async def _call_openroute(self, messages: List[Dict[str, str]]) -> str:
        """Call the OpenRoute API asynchronously."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "MCP-Chatbot"
        }
        
        payload = {
            "messages": messages,
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False
        }
        
        session = await self._ensure_session()
        async with session.post(url, headers=headers, json=payload) as response:
            data = await response.json()
            
            # Extract token usage (OpenRoute follows the OpenAI format)
            if "usage" in data:
                self.last_token_usage.update(
                    prompt_tokens=data["usage"].get("prompt_tokens", 0),
                    completion_tokens=data["usage"].get("completion_tokens", 0),
                    total_tokens=data["usage"].get("total_tokens", 0)
                )
            else:
                # If no token info provided, estimate based on content
                prompt_text = "\n".join([msg.get("content", "") for msg in messages])
                response_text = data['choices'][0]['message']['content']
                
                prompt_tokens = self.estimate_tokens(prompt_text)
                completion_tokens = self.estimate_tokens(response_text)
                
                self.last_token_usage.update(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            
            return data['choices'][0]['message']['content']

    async def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Call the Ollama API asynchronously."""
        # Construct URL from configured host
        ollama_host = self.config.ollama_host or "http://localhost:11434"
        url = f"{ollama_host.rstrip('/')}/api/chat"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": 0.7,
                "num_predict": 4096
            }
        }
        
        # session = await self._ensure_session()
        # async with session.post(url, headers=headers, json=payload) as response:
        #     data = await response.json()
        #     if "message" in data:
        #         return data["message"]["content"]
        #     return "No text content returned from Ollama API"
        session = await self._ensure_session()
        async with session.post(url, headers=headers, json=payload) as response:
            data = await response.json()
            
            response_text = ""
            if "message" in data:
                response_text = data["message"]["content"]
            else:
                response_text = "No text content returned from Ollama API"
            
            # Ollama may provide token metrics in these fields
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            
            if prompt_tokens > 0 or completion_tokens > 0:
                # Use actual values if provided
                self.last_token_usage.update(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            else:
                # Estimate using our method if not provided
                prompt_text = "\n".join([msg.get("content", "") for msg in messages])
                
                prompt_tokens = self.estimate_tokens(prompt_text)
                completion_tokens = self.estimate_tokens(response_text)
                
                self.last_token_usage.update(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
                
            return response_text

    def safe_json_serialize(self, obj: Any) -> Any:
        """Safely serialize objects to JSON, handling cases like NaN, Infinity, etc."""
        if isinstance(obj, (str, int, bool, type(None))):
            return obj
        elif isinstance(obj, float):
            # Handle special floating point values
            if not (float("-inf") < obj < float("inf")):
                return str(obj)  # Convert Inf and NaN to strings
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self.safe_json_serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self.safe_json_serialize(v) for k, v in obj.items()}
        else:
            # For other objects, convert to string representation
            return str(obj)

    async def cleanup(self) -> None:
        """Clean up client resources."""
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session if open
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: List[Server], llm_client: LLMClient) -> None:
        self.servers: List[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._model_refresh_interval: int = 300  # Refresh models every 5 minutes
        self._model_refresh_task: Optional[asyncio.Task] = None
        self._max_history_length: int = llm_client.config.message_history_limit


    async def list_resources(self) -> List[Dict[str, str]]:
        """List available resources from all servers with reconnection logic.
        
        Returns:
            A list of resources with their metadata.
        """
        all_resources = []
        for server in self.servers:
            try:
                if server.state != ServerState.CONNECTED:
                    reconnected = await server.initialize()
                    if not reconnected:
                        continue
                        
                resources = await server.session.list_resources()
                if resources:
                    all_resources.extend(resources)
                    
            except Exception as e:
                logging.error(f"Error listing resources from server {server.name}: {e}")

                
        return all_resources

    async def list_prompts(self) -> List[Dict[str, Any]]:
        """List available prompts from all servers with reconnection logic.
        
        Returns:
            A list of available prompts with their metadata.
        """
        all_prompts = []
        for server in self.servers:
            try:
                if server.state != ServerState.CONNECTED:
                    reconnected = await server.initialize()
                    if not reconnected:
                        continue
                        
                prompts = await server.session.list_prompts()
                if prompts:
                    all_prompts.extend(prompts)
                    
            except Exception as e:
                logging.error(f"Error listing prompts from server {server.name}: {e}")
                
        return all_prompts

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource from its URI.
        
        Args:
            uri: The URI of the resource to read.
            
        Returns:
            The content of the resource.
            
        Raises:
            RuntimeError: If resource reading fails.
        """
        for server in self.servers:
            if server.state != ServerState.CONNECTED:
                continue
                
            try:
                result = await server.session.read_resource(uri)
                return result
            except Exception as e:
                logging.warning(f"Failed to read resource {uri} from {server.name}: {e}")
                
        raise RuntimeError(f"Failed to read resource: {uri}")

    async def get_prompt(self, name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get a prompt by name with optional arguments.
        
        Args:
            name: The name of the prompt.
            arguments: Optional arguments for the prompt.
            
        Returns:
            Prompt result including messages.
            
        Raises:
            RuntimeError: If no server has the prompt or execution fails.
        """
        for server in self.servers:
            if server.state != ServerState.CONNECTED:
                continue
                
            try:
                prompts = await server.session.list_prompts()
                if any(prompt.name == name for prompt in prompts):
                    result = await server.session.get_prompt(name, arguments or {})
                    return result
            except Exception as e:
                logging.warning(f"Failed to get prompt {name} from {server.name}: {e}")
                
        raise RuntimeError(f"No server found with prompt: {name}")
    
    async def display_resources_list(self) -> None:
        """Display a formatted list of available resources."""
        async def fetch_and_display():
            resources = await self.list_resources()
            
            if not resources:
                print("\n  No resources available. Servers may not be connected.")
                return
                
            print("\n===== Available MCP Resources =====")
            
            # Group resources by server
            server_resources = {}
            
            if isinstance(resources, list):
                # 查找键为 'resources' 的元组
                for item in resources:
                    if isinstance(item, tuple) and len(item) >= 2 and item[0] == 'resources':
                        actual_resources = item[1]
                        
                        # 处理提取出的 resources
                        for resource in actual_resources:
                            # 尝试各种可能的方式获取 server 属性
                            if hasattr(resource, 'name'):
                                server_name = resource.name
                            elif isinstance(resource, dict) and 'server' in resource:
                                server_name = resource['server']
                            elif hasattr(resource, 'get'):
                                server_name = resource.get('server', 'Unknown')
                            else:
                                server_name = 'Unknown'
                            
                            if server_name not in server_resources:
                                server_resources[server_name] = []
                            server_resources[server_name].append(resource)
                        break  # 找到并处理了 resources，可以退出循环
                
                # 如果没有找到特定结构，尝试处理直接的资源列表
                if not server_resources and resources:
                    for resource in resources:
                        # 尝试各种可能的方式获取 server 属性
                        if hasattr(resource, 'name'):
                            server_name = resource.name
                        elif isinstance(resource, dict) and 'server' in resource:
                            server_name = resource['server']
                        elif hasattr(resource, 'get'):
                            server_name = resource.get('server', 'Unknown')
                        else:
                            server_name = 'Unknown'
                        
                        if server_name not in server_resources:
                            server_resources[server_name] = []
                        server_resources[server_name].append(resource)
            
            # Display resources by server
            for server_name, res_list in server_resources.items():
                print(f"\n{server_name} Server Resources:")
                for resource in res_list:
                    if hasattr(resource, 'uri'):
                        uri = resource.uri
                        name = resource.name
                        description = resource.description
                    else:
                        uri = resource.get("uri", "Unknown URI") if hasattr(resource, 'get') else "Unknown URI"
                        name = resource.get("name", "Unnamed Resource") if hasattr(resource, 'get') else "Unnamed Resource"
                        description = resource.get("description", "No description") if hasattr(resource, 'get') else "No description"
                    
                    print(f"  • {name} ({uri})")
                    print(f"    {description}")
                    
            print(f"\nTotal available resources: {len(server_resources)}")
        
        await asyncio.create_task(fetch_and_display())
    async def display_prompts_list(self) -> None:
        """Display a formatted list of available prompts."""
        async def fetch_and_display():
            prompts = await self.list_prompts()
            
            if not prompts:
                print("\n  No prompts available. Servers may not be connected.")
                return
                
            print("\n===== Available MCP Prompts =====")
            
            # Group prompts by server
            server_prompts = {}
            # for prompt in prompts:
            #     server_name = prompt.get("server", "Unknown")
            #     if server_name not in server_prompts:
            #         server_prompts[server_name] = []
            #     server_prompts[server_name].append(prompt)
            
            if isinstance(prompts, list):
                # 查找键为 'prompts' 的元组
                for item in prompts:
                    if isinstance(item, tuple) and len(item) >= 2 and item[0] == 'prompts':
                        actual_prompts = item[1]
                        
                        # 处理提取出的 prompts
                        for prompt in actual_prompts:
                            # 尝试各种可能的方式获取 server 属性
                            if hasattr(prompt, 'name'):
                                server_name = prompt.name
                            elif isinstance(prompt, dict) and 'server' in prompt:
                                server_name = prompt['server']
                            elif hasattr(prompt, 'get'):
                                server_name = prompt.get('server', 'Unknown')
                            else:
                                server_name = 'Unknown'
                            
                            if server_name not in server_prompts:
                                server_prompts[server_name] = []
                            server_prompts[server_name].append(prompt)
            
            # Display prompts by server
            for server_name, prompt_list in server_prompts.items():
                print(f"\n{server_name} Server Prompts:")
                for prompt in prompt_list:
                    name = prompt.name
                    description = prompt.description
                    arguments = prompt.arguments
                    print(f"  • {name}: {description}")
                    if arguments:
                        print(f"    Arguments:")
                        for arg in arguments:
                            arg_name = arg.name
                            arg_desc = arg.description
                            required = " (required)" if arg.required else ""
                            print(f"      - {arg_name}: {arg_desc}{required}")
                    
            print(f"\nTotal available prompts: {len(server_prompts)}")
        
        await asyncio.create_task(fetch_and_display())

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))
        
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")
        
        # Clean up LLM client
        await self.llm_client.cleanup()

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools, read resources, or get prompts if needed.
        
        Args:
            llm_response: The response from the LLM.
            
        Returns:
            The result of tool execution or the original response.
        """
        import json
        import re
        
        try:
            # Check if response is wrapped in markdown code block
            json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            match = re.search(json_pattern, llm_response)
            
            if match:
                # Extract the JSON from the markdown
                json_str = match.group(1)
                try:
                    parsed_response = json.loads(json_str)
                except json.JSONDecodeError:
                    return llm_response
            else:
                # Try to parse as direct JSON
                try:
                    parsed_response = json.loads(llm_response)
                except json.JSONDecodeError:
                    return llm_response
                    
            # Handle tool calls
            if "tool" in parsed_response and "arguments" in parsed_response:
                tool_name = parsed_response['tool']
                print(f"\n> Executing tool: {tool_name}")
                print(f"> With arguments: {json.dumps(parsed_response['arguments'], indent=2)}")
                logging.info(f"Executing tool: {parsed_response['tool']}")
                logging.info(f"With arguments: {parsed_response['arguments']}")
                
                # Try to find the tool across all servers
                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_name for tool in tools):
                        try:
                            result = await server.execute_tool(tool_name, parsed_response["arguments"])
                            
                            # Safe handling of progress calculations
                            if isinstance(result, dict) and 'progress' in result and 'total' in result:
                                progress = result['progress']
                                total = result['total']
                                
                                # Avoid division by zero
                                if total > 0:
                                    logging.info(f"Progress: {progress}/{total} ({(progress/total)*100:.1f}%)")
                                else:
                                    logging.info(f"Progress: {progress}/{total}")
                            
                            # Format and display the result
                            result_str = str(result)
                            if len(result_str) > 1000:
                                print(f"> Tool result (truncated): {result_str[:1000]}...")
                            else:
                                print(f"> Tool result: {result_str}")
                                
                            print(f"> Tool execution completed: {tool_name}")
                            
                            # Serialize result for LLM
                            serialized_result = self.llm_client.safe_json_serialize(result)
                            return f"Tool execution result: {serialized_result}"
                            
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logging.error(error_msg)
                            
                            # Try reconnecting the server
                            await server.initialize()
                            
                            # Retry the tool execution once more after reconnection
                            try:
                                result = await server.execute_tool(tool_name, parsed_response["arguments"])
                                serialized_result = self.llm_client.safe_json_serialize(result)
                                return f"Tool execution result: {serialized_result}"
                            except Exception as retry_error:
                                return f"Error executing tool after reconnection: {str(retry_error)}"
                
                return f"No server found with tool: {tool_name}"
            
            # Handle resource reads
            elif "resource" in parsed_response and "uri" in parsed_response["resource"]:
                resource_uri = parsed_response["resource"]["uri"]
                print(f"\n> Reading resource: {resource_uri}")
                logging.info(f"Reading resource: {resource_uri}")
                
                try:
                    resource_result = await self.read_resource(resource_uri)
                    
                    # Format and display the result
                    result_str = str(resource_result)
                    if len(result_str) > 1000:
                        print(f"> Resource content (truncated): {result_str[:1000]}...")
                    else:
                        print(f"> Resource content: {result_str}")
                    
                    print(f"> Resource reading completed: {resource_uri}")
                    
                    # Serialize result for LLM
                    serialized_result = self.llm_client.safe_json_serialize(resource_result)
                    return f"Resource content: {serialized_result}"
                    
                except Exception as e:
                    error_msg = f"Error reading resource: {str(e)}"
                    logging.error(error_msg)
                    return error_msg
            
            # Handle prompt execution
            elif "prompt" in parsed_response and "name" in parsed_response["prompt"]:
                prompt_name = parsed_response["prompt"]["name"]
                prompt_arguments = parsed_response["prompt"].get("arguments", {})
                print(f"\n> Executing prompt: {prompt_name}")
                if prompt_arguments:
                    print(f"> With arguments: {json.dumps(prompt_arguments, indent=2)}")
                logging.info(f"Executing prompt: {prompt_name}")
                logging.info(f"With arguments: {prompt_arguments}")
                
                try:
                    prompt_result = await self.get_prompt(prompt_name, prompt_arguments)
                    
                    # Format and display the result
                    result_str = str(prompt_result)
                    if len(result_str) > 1000:
                        print(f"> Prompt result (truncated): {result_str[:1000]}...")
                    else:
                        print(f"> Prompt result: {result_str}")
                    
                    print(f"> Prompt execution completed: {prompt_name}")
                    
                    # Serialize result for LLM
                    serialized_result = self.llm_client.safe_json_serialize(prompt_result)
                    return f"Prompt result: {serialized_result}"
                    
                except Exception as e:
                    error_msg = f"Error executing prompt: {str(e)}"
                    logging.error(error_msg)
                    return error_msg
                    
            return llm_response
        except Exception as e:
            logging.error(f"Error processing LLM response: {e}")
            return llm_response

    def display_llm_list(self) -> None:
        """Display a formatted list of available LLM providers and models."""
        print("\n===== Available LLM Providers and Models =====")
        
        for provider in self.llm_client.PROVIDER_CONFIGS.keys():
            # Check if API key is available (except for Ollama which doesn't need one)
            api_key_status = "✓" if provider == "ollama" or self.llm_client.config.get_api_key(provider) else "✗"
            health_status = "🟢" if self.llm_client.provider_health.get(provider, False) else "🔴"
                
            if provider == self.llm_client.provider:
                print(f"\n🔹 {provider.upper()} {api_key_status} {health_status}")
            else:
                print(f"\n• {provider.upper()} {api_key_status} {health_status}")
            
            # Get models for this provider
            models = self.llm_client.available_models.get(provider, [])
            
            # Only show models if the provider is healthy
            if self.llm_client.provider_health.get(provider, False) and models:
                # Print models with the current one highlighted (limit to 10)
                for model in models[:10]:
                    if provider == self.llm_client.provider and model == self.llm_client.model:
                        print(f"   ➜ {model}")
                    else:
                        print(f"     {model}")
                        
                # If there are more models, indicate this
                remaining = len(models) - 10
                if remaining > 0:
                    print(f"     ... and {remaining} more models")
            else:
                # No models available - show error
                print("   No models available - API connection failed")
            
            # If no API key is available, add a note
            if api_key_status == "✗" and provider != "ollama":
                print("     (Add API key in .env file to use this provider)")
        
        print("\nUse '/switch <provider> <model>' to change the LLM")
        print("Example: /switch openai gpt-4o")
        print("\nAPI Key Status: ✓ = configured, ✗ = missing")
        # Add after line 2254 (after "API Key Status" legend)
        print("Health Status: 🟢 = healthy, 🔴 = unhealthy")
    
    def _manage_message_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Maintain message history within limits to prevent token overflow.
        
        Args:
            messages: Current message history
            
        Returns:
            Updated message history within limits
        """
        # Always keep the system message(s)
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        non_system_messages = [msg for msg in messages if msg["role"] != "system"]
        
        # If we're over the limit, trim the non-system messages
        if len(non_system_messages) > self._max_history_length:
            # Keep only the most recent messages
            trimmed_messages = non_system_messages[-self._max_history_length:]
            logging.info(f"Trimmed message history from {len(non_system_messages)} to {len(trimmed_messages)} messages")
            
            # Prepend system messages to maintain context
            return system_messages + trimmed_messages
            
        return messages
    
    async def _worker(self, messages: List[Dict[str, str]]) -> None:
        """Background worker to process messages asynchronously.
        
        Args:
            messages: The current message history
        """
        # Use a deep copy to avoid shared references
        messages = deepcopy(messages)
        
        while True:
            try:
                # Get the next user input from the queue
                user_input = await self.message_queue.get()
                
                if user_input == "__TERMINATE__":
                    break
                    
                # Add user message to history
                messages.append({"role": "user", "content": user_input})
                
                # Apply message history management
                messages = self._manage_message_history(messages)
                
                # Track the start time and input token count (estimated)
                start_time = time.time()
                input_token_count = sum(len(msg["content"].split()) * 1.3 for msg in messages)
                
                # Get LLM response
                llm_response, token_usage = await self.llm_client.get_response(messages)
                
                # Calculate time taken and output token count (estimated)
                time_taken = time.time() - start_time
                output_token_count = len(llm_response.split()) * 1.3
                
                # Log stats
                stats_info = f"\n[Model: {self.llm_client.provider}/{self.llm_client.model}] {token_usage} [Time: {time_taken:.2f}s]"
                logging.info(stats_info)
                
                logging.info("\nAssistant: %s", llm_response)

                # Process the response and execute any tools
                result = await self.process_llm_response(llm_response)
                
                # If we got a result from a tool, get a final response
                if result != llm_response:
                    messages.append({"role": "assistant", "content": llm_response})
                    messages.append({"role": "system", "content": result})
                    
                    # Apply message history management again
                    messages = self._manage_message_history(messages)
                    
                    # Track start time for final response
                    tool_start_time = time.time()
                    
                    final_response, final_token_usage = await self.llm_client.get_response(messages)
                    
                    # Calculate time and tokens for final response
                    tool_time_taken = time.time() - tool_start_time
                    final_output_tokens = len(final_response.split()) * 1.3
                    
                    # Log final stats
                    final_stats = f"\n[Model: {self.llm_client.provider}/{self.llm_client.model}] {final_token_usage} [Time: {tool_time_taken:.2f}s]"
                    logging.info(final_stats)
                    
                    logging.info("\nFinal response: %s", final_response)
                    messages.append({"role": "assistant", "content": final_response})
                    
                    # Put the final response with stats in the queue
                    await self.response_queue.put(f"{final_response}\n{final_stats}")
                else:
                    # Add the LLM response to history
                    messages.append({"role": "assistant", "content": llm_response})
                    
                    # Put the response with stats in the queue
                    await self.response_queue.put(f"{llm_response}\n{stats_info}")
                
                # Mark this task as done
                self.message_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in worker: {e}")
                await self.response_queue.put(f"An error occurred: {str(e)}")
                self.message_queue.task_done()

    async def start_model_refresh(self) -> None:
        """Start a periodic task to refresh models for all providers."""
        if self._model_refresh_task and not self._model_refresh_task.done():
            self._model_refresh_task.cancel()
            
        self._model_refresh_task = asyncio.create_task(self._model_refresh_loop())
        
    async def _model_refresh_loop(self) -> None:
        """Periodically refresh the model lists for all providers."""
        while True:
            try:
                # Wait for the refresh interval
                await asyncio.sleep(self._model_refresh_interval)
                
                logging.info("Refreshing model lists for all providers...")
                
                # Create fetch tasks for all providers
                fetch_tasks = []
                for provider_name in self.llm_client.PROVIDER_CONFIGS.keys():
                    # Only try to fetch if we have an API key (except for Ollama)
                    if provider_name == "ollama" or self.llm_client.config.get_api_key(provider_name) is not None:
                        task = asyncio.create_task(self.llm_client._fetch_provider_models(provider_name))
                        fetch_tasks.append(task)
                
                # Wait for all fetches to complete (don't raise exceptions)
                if fetch_tasks:
                    await asyncio.gather(*fetch_tasks, return_exceptions=True)
                    
                logging.info("Model refresh complete")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in model refresh loop: {e}")
    
    async def start(self) -> None:
        """Main chat session handler with improved async flow."""
        try:
            # Initialize all servers in parallel
            init_tasks = []
            for server in self.servers:
                init_tasks.append(asyncio.create_task(server.initialize()))
            
            # Initialize LLM client in parallel
            llm_init_task = asyncio.create_task(self.llm_client.initialize())
            
            server_init_results = []
            for server in self.servers:
                try:
                    success = await server.initialize()
                    server_init_results.append((server.name, success))
                except Exception as e:
                    logging.error(f"Failed to initialize server {server.name}: {e}")
                    server_init_results.append((server.name, False))
            try:
                await self.llm_client.initialize()
            except Exception as e:
                logging.error(f"Failed to initialize LLM client: {e}")
                print(f"\nError initializing LLM client: {e}")
                print("Continuing with reduced functionality...")    
                
            successful_servers = [name for name, success in server_init_results if success]
            failed_servers = [name for name, success in server_init_results if not success]

            if successful_servers:
                logging.info(f"Successfully initialized servers: {', '.join(successful_servers)}")
            else:
                logging.warning("No servers were successfully initialized!")

            if failed_servers:
                logging.warning(f"Failed to initialize servers: {', '.join(failed_servers)}")
                print(f"\nWarning: Some MCP servers failed to initialize: {', '.join(failed_servers)}")
                print("You can continue with reduced functionality.")
            
            # Wait for all initializations to complete
            await asyncio.gather(*init_tasks, llm_init_task)
            
            # Start periodic model refreshing
            await self.start_model_refresh()
            
            # Collect all tools from all servers
            all_tools = []
            tool_tasks = []
            for server in self.servers:
                task = asyncio.create_task(server.list_tools())
                tool_tasks.append(task)
            
            # Wait for all tool lists and combine them
            tool_results = await asyncio.gather(*tool_tasks)
            for tools in tool_results:
                all_tools.extend(tools)
            
            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
            
            # ---------- RESOURCES ----------
            # Collect all resources - using the exact same logic as display_resources_list
            resources = await self.list_resources()
            all_resources = []
            
            if isinstance(resources, list):
                # Look for tuples with 'resources' as first element
                for item in resources:
                    if isinstance(item, tuple) and len(item) >= 2 and item[0] == 'resources':
                        actual_resources = item[1]
                        for resource in actual_resources:
                            all_resources.append(resource)
                        break  # Found and processed resources, exit the loop
                
                # If no specific tuple structure found, try processing as direct resource list
                if not all_resources and resources:
                    all_resources = resources
            
            logging.info(f"Found {len(all_resources)} resources to include in system message")
            
            # Format resources for LLM
            resources_description = ""
            if all_resources:
                resources_description = "Available Resources:\n"
                for res in all_resources:
                    if hasattr(res, 'uri'):
                        uri = res.uri
                        name = res.name
                        description = res.description
                    else:
                        uri = res.get("uri", "Unknown URI") if hasattr(res, 'get') else "Unknown URI"
                        name = res.get("name", "Unnamed Resource") if hasattr(res, 'get') else "Unnamed Resource"
                        description = res.get("description", "No description") if hasattr(res, 'get') else "No description"
                    
                    resources_description += f"- {name} ({uri}): {description}\n"
            
            # ---------- PROMPTS ----------
            # Collect all prompts - using the exact same logic as display_prompts_list
            prompts = await self.list_prompts()
            all_prompts = []
            
            if isinstance(prompts, list):
                # Look for tuples with 'prompts' as first element
                for item in prompts:
                    if isinstance(item, tuple) and len(item) >= 2 and item[0] == 'prompts':
                        actual_prompts = item[1]
                        for prompt in actual_prompts:
                            all_prompts.append(prompt)
                        break  # Found and processed prompts, exit the loop
                
                # If no specific tuple structure found, try processing as direct prompt list
                if not all_prompts and prompts:
                    all_prompts = prompts
            
            logging.info(f"Found {len(all_prompts)} prompts to include in system message")
            
            # Format prompts for LLM
            prompts_description = ""
            if all_prompts:
                prompts_description = "Available Prompts:\n"
                for prompt in all_prompts:
                    if hasattr(prompt, 'name'):
                        name = prompt.name
                        description = prompt.description
                        arguments = prompt.arguments
                    else:
                        name = prompt.get("name", "Unnamed Prompt") if hasattr(prompt, 'get') else "Unnamed Prompt"
                        description = prompt.get("description", "No description") if hasattr(prompt, 'get') else "No description"
                        arguments = prompt.get("arguments", []) if hasattr(prompt, 'get') else []
                    
                    prompts_description += f"- {name}: {description}\n"
                    
                    # Add arguments if available
                    if arguments:
                        prompts_description += f"  Arguments:\n"
                        for arg in arguments:
                            if hasattr(arg, 'name'):
                                arg_name = arg.name
                                arg_desc = arg.description
                                required = " (required)" if arg.required else ""
                            else:
                                arg_name = arg.get("name", "unknown") if hasattr(arg, 'get') else "unknown"
                                arg_desc = arg.get("description", "No description") if hasattr(arg, 'get') else "No description"
                                required = " (required)" if (hasattr(arg, 'get') and arg.get('required')) else ""
                            
                            prompts_description += f"    - {arg_name}: {arg_desc}{required}\n"
            
            system_message = f"""You are a helpful assistant with access to these tools: 

    {tools_description}

    {resources_description}

    {prompts_description}

    Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.

    IMPORTANT: When you need to use a tool, you must ONLY respond with the exact JSON object format below, nothing else:
    {{
        "tool": "tool-name",
        "arguments": {{
            "argument-name": "value"
        }}
    }}

    After receiving a tool's response:
    1. Transform the raw data into a natural, conversational response
    2. Keep responses concise but informative
    3. Focus on the most relevant information
    4. Use appropriate context from the user's question
    5. Avoid simply repeating the raw data

    Please use only the tools that are explicitly defined above."""

            messages = [
                {
                    "role": "system",
                    "content": system_message
                }
            ]
            
            # Show current LLM provider and model at startup
            print(f"\nCurrent LLM: {self.llm_client.provider.upper()} - {self.llm_client.model}")
            print("Type '/llm' to see all available providers and models")
            print("Type '/help' for more commands\n")

            # Start the worker task - use a deep copy to prevent shared state issues
            self._worker_task = asyncio.create_task(self._worker(deepcopy(messages)))

            while True:
                try:
                    # Get user input
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("You: ").strip()
                    )
                    
                    if user_input.lower() in ['quit', 'exit']:
                        logging.info("\nExiting...")
                        break
                    
                    # Check for provider switch commands
                    if user_input.startswith("/switch "):
                        parts = user_input.split()
                        if len(parts) >= 3:
                            provider = parts[1].lower()
                            model = parts[2]
                            
                            try:
                                await self.llm_client.change_provider(provider, model)
                                print(f"\nSwitched to {provider.upper()} with model {self.llm_client.model}")
                                continue
                            except ValueError as e:
                                print(f"\nError: {str(e)}")
                                continue
                        else:
                            print("\nUsage: /switch <provider> <model>")
                            print("Example: /switch openai gpt-4o")
                            print("Type '/llm' to see available providers and models")
                            continue
                    
                    elif user_input.lower() == "/llm":
                        self.display_llm_list()
                        continue
                    
                    elif user_input.lower() == "/refresh":
                        print("\nRefreshing model lists for all providers...")
                        refresh_tasks = []
                        for provider_name in self.llm_client.PROVIDER_CONFIGS.keys():
                            if provider_name == "ollama" or self.llm_client.config.get_api_key(provider_name) is not None:
                                task = asyncio.create_task(
                                    self.llm_client._fetch_provider_models(provider_name)
                                )
                                refresh_tasks.append(task)
                        
                        if refresh_tasks:
                            await asyncio.gather(*refresh_tasks, return_exceptions=True)
                        
                        print("Model refresh complete")
                        self.display_llm_list()
                        continue
                    elif user_input.lower() == "/tools":
                        print("\n===== Available MCP Tools =====")
                        all_tools = []
                        for server in self.servers:
                            if server.state == ServerState.CONNECTED:
                                tools = await server.list_tools()
                                if tools:
                                    print(f"\n{server.name} Server Tools:")
                                    for tool in tools:
                                        print(f"  • {tool.name} - {tool.description}")
                                        all_tools.append(tool)
                        
                        if not all_tools:
                            print("  No tools available. Servers may not be connected.")
                        print(f"\nTotal available tools: {len(all_tools)}")
                        continue
                    
                    # Add support for /resources command
                    elif user_input.lower() == "/resources":
                        await self.display_resources_list()
                        continue
                            
                    
                    # Add support for /prompts command
                    elif user_input.lower() == "/prompts":
                        await self.display_prompts_list()
                        continue

                    elif user_input.lower() == "/now":
                        print(f"\nCurrent LLM: {self.llm_client.provider.upper()} - {self.llm_client.model}")
                        if hasattr(self.llm_client, 'last_token_usage'):
                            print(f"Last token usage: {self.llm_client.last_token_usage}")
                        print(f"Provider health status: {'✅ Healthy' if self.llm_client.provider_health.get(self.llm_client.provider, False) else '❌ Unhealthy'}")
                        continue
                    elif user_input.lower() == "/help":
                        print("\n===== Available Commands =====")
                        print("  /llm                      - Show available LLM providers and models")
                        print("  /switch <provider> <model> - Switch to a different LLM")
                        print("  /refresh                  - Refresh model lists for all providers")
                        print("  /tools                    - Show available MCP tools")
                        print("  /resources                - Show available MCP resources")  
                        print("  /prompts                  - Show available MCP prompts")    
                        print("  /now                      - Show current LLM details")
                        print("  /help                     - Show this help message")
                        print("  quit or exit              - Exit the chat")
                        continue

                    # Put the user input in the queue for processing
                    await self.message_queue.put(user_input)
                    
                    # Wait for the response
                    response = await self.response_queue.get()
                    print(f"Assistant: {response}")
                    self.response_queue.task_done()

                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break
        
        finally:
            # Signal the worker to terminate
            if self._worker_task and not self._worker_task.done():
                await self.message_queue.put("__TERMINATE__")
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel model refresh task
            if self._model_refresh_task and not self._model_refresh_task.done():
                self._model_refresh_task.cancel()
                try:
                    await self._model_refresh_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up all resources
            await self.cleanup_servers()


class ConfigModel(pydantic.BaseModel):
    """Pydantic model for validating server configuration."""
    mcpServers: Dict[str, Dict[str, Any]]


async def main() -> None:
    """Initialize and run the chat session."""
    try:
        config = Configuration()
        
        # Load and validate server configuration
        try:
            server_config_dict = config.load_config('servers_config.json')
            server_config = ConfigModel(**server_config_dict)
        except (ConfigurationError, pydantic.ValidationError) as e:
            logging.error(f"Configuration error: {e}")
            return
        
        # Create servers with the global config
        servers = [Server(name, srv_config, config) for name, srv_config in server_config.mcpServers.items()]
        
        # Create and initialize LLM client
        llm_client = LLMClient(config)
        
        # Create and start chat session
        chat_session = ChatSession(servers, llm_client)
        await chat_session.start()
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())