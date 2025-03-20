import asyncio
import json
import logging
import threading
import queue
import time
import os
import sys
from flask import Flask, request, render_template, jsonify, send_from_directory
from typing import Dict, List, Any, Optional

# Import components from ChatMCP
from ChatMCP import (
    Configuration, Server, LLMClient, 
    ChatSession, ConfigurationError, TokenUsage, ServerState
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='flask_chatbot.log',
    filemode='a'
)

# Global queues for communication
input_queue = queue.Queue()  # Web UI → Async thread
output_queue = queue.Queue()  # Async thread → Web UI
status_queue = queue.Queue()  # Status updates
metadata_queue = queue.Queue()  # For tools, resources, prompts info

# Global state
current_provider = ""
current_model = ""
available_tools = []
available_resources = []
available_prompts = []
token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
provider_models = {}
provider_health = {}
is_initialized = False  # Flag to track initialization status
chat_history = []  # Store chat history

# Flag to signal shutdown
shutdown_event = threading.Event()

# Create the Flask app
app = Flask(__name__)

def run_async_loop(coro):
    """Create a new event loop in the current thread and run a coroutine."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def async_init():
    """Initialize components asynchronously in a dedicated thread."""
    run_async_loop(initialize_components())

async def initialize_components():
    """Initialize configuration, servers, and LLM client."""
    global current_provider, current_model, provider_models, provider_health, is_initialized
    
    try:
        status_queue.put("Initializing configuration...")
        config = Configuration()
        
        # Load server configuration
        try:
            server_config_dict = config.load_config('servers_config.json')
        except Exception as e:
            logging.error(f"Configuration error: {e}")
            status_queue.put(f"Error loading configuration: {e}")
            return
        
        # Create servers with the global config
        status_queue.put("Creating servers...")
        servers = []
        for name, srv_config in server_config_dict["mcpServers"].items():
            servers.append(Server(name, srv_config, config))
        
        # Create and initialize LLM client
        status_queue.put("Initializing LLM client...")
        llm_client = LLMClient(config)
        await llm_client.initialize()
        
        # Initialize servers in parallel
        status_queue.put("Connecting to MCP servers...")
        init_tasks = []
        for server in servers:
            init_tasks.append(asyncio.create_task(server.initialize()))
        
        if init_tasks:
            await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Create chat session
        status_queue.put("Creating chat session...")
        chat_session = ChatSession(servers, llm_client)
        
        # Fetch available tools, resources, and prompts
        status_queue.put("Fetching tools and resources...")
        await fetch_metadata(chat_session)
        
        # Update global variables with LLM information
        current_provider = llm_client.provider
        current_model = llm_client.model
        provider_models = llm_client.available_models
        
        # Get provider health status
        for provider, status in llm_client.provider_health.items():
            provider_health[provider] = "🟢 Healthy" if status else "🔴 Unhealthy"
        
        # Set initialization flag
        is_initialized = True
        status_queue.put(f"Initialization complete. Using {current_provider}/{current_model}")
        
        # Generate metadata HTML
        metadata_html = generate_metadata_html()
        metadata_queue.put(metadata_html)
        
        # Start processing messages
        await process_messages(chat_session)
        
    except Exception as e:
        logging.error(f"Initialization error: {e}")
        status_queue.put(f"Error during initialization: {e}")

async def fetch_metadata(chat_session):
    """Fetch tools, resources, and prompts."""
    global available_tools, available_resources, available_prompts
    
    # Fetch tools
    all_tools = []
    for server in chat_session.servers:
        if server.state != ServerState.CONNECTED:
            continue
            
        try:
            tools = await server.list_tools()
            for tool in tools:
                all_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "server": server.name
                })
        except Exception as e:
            logging.error(f"Error fetching tools from {server.name}: {e}")
    
    available_tools = all_tools
    
    # Fetch resources
    try:
        resources = await chat_session.list_resources()
        all_resources = []
        
        if isinstance(resources, list):
            for item in resources:
                if isinstance(item, tuple) and len(item) >= 2 and item[0] == 'resources':
                    for resource in item[1]:
                        if hasattr(resource, 'uri'):
                            all_resources.append({
                                "name": resource.name,
                                "uri": resource.uri,
                                "description": resource.description
                            })
        
        available_resources = all_resources
    except Exception as e:
        logging.error(f"Error fetching resources: {e}")
    
    # Fetch prompts
    try:
        prompts = await chat_session.list_prompts()
        all_prompts = []
        
        if isinstance(prompts, list):
            for item in prompts:
                if isinstance(item, tuple) and len(item) >= 2 and item[0] == 'prompts':
                    for prompt in item[1]:
                        if hasattr(prompt, 'name'):
                            all_prompts.append({
                                "name": prompt.name,
                                "description": prompt.description
                            })
        
        available_prompts = all_prompts
    except Exception as e:
        logging.error(f"Error fetching prompts: {e}")

def generate_metadata_html():
    """Generate HTML for tools, resources, and prompts."""
    tools_html = "<h3>Available Tools</h3>"
    if available_tools:
        for tool in available_tools:
            tools_html += f"""
            <div style="margin-bottom: 15px;">
                <h4>{tool['name']}</h4>
                <p>{tool['description']}</p>
                <p><em>Server: {tool['server']}</em></p>
            </div>
            """
    else:
        tools_html += "<p>No tools available</p>"
    
    resources_html = "<h3>Available Resources</h3>"
    if available_resources:
        for res in available_resources:
            resources_html += f"""
            <div style="margin-bottom: 15px;">
                <h4>{res['name']}</h4>
                <p>{res['description']}</p>
                <p><em>URI: {res['uri']}</em></p>
            </div>
            """
    else:
        resources_html += "<p>No resources available</p>"
    
    prompts_html = "<h3>Available Prompts</h3>"
    if available_prompts:
        for prompt in available_prompts:
            prompts_html += f"""
            <div style="margin-bottom: 15px;">
                <h4>{prompt['name']}</h4>
                <p>{prompt['description']}</p>
            </div>
            """
    else:
        prompts_html += "<p>No prompts available</p>"
    
    return tools_html + resources_html + prompts_html

async def process_messages(chat_session):
    """Process messages from the input queue and generate responses."""
    # Create a system message with available tools, resources, and prompts
    all_tools = []
    for server in chat_session.servers:
        try:
            tools = await server.list_tools()
            all_tools.extend(tools)
        except Exception as e:
            logging.error(f"Error fetching tools from {server.name}: {e}")
    
    tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
    
    # Prepare resource descriptions
    resources_description = ""
    if available_resources:
        resources_description = "Available Resources:\n"
        for res in available_resources:
            resources_description += f"- {res['name']} ({res['uri']}): {res['description']}\n"
    
    # Prepare prompt descriptions
    prompts_description = ""
    if available_prompts:
        prompts_description = "Available Prompts:\n"
        for prompt in available_prompts:
            prompts_description += f"- {prompt['name']}: {prompt['description']}\n"
    
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
    
    while not shutdown_event.is_set():
        try:
            # Check if there's a message in the queue (non-blocking)
            try:
                user_message = input_queue.get(block=False)
                
                if user_message.startswith("/switch "):
                    # Handle provider/model switching
                    parts = user_message.split()
                    if len(parts) >= 3:
                        provider = parts[1].lower()
                        model = parts[2]
                        
                        try:
                            status_queue.put(f"Switching to {provider}/{model}...")
                            await chat_session.llm_client.change_provider(provider, model)
                            
                            # Update globals
                            global current_provider, current_model
                            current_provider = provider
                            current_model = model
                            
                            output_queue.put({
                                "type": "switch_result",
                                "success": True,
                                "message": f"Switched to {provider.upper()} with model {model}"
                            })
                        except Exception as e:
                            logging.error(f"Error switching provider/model: {e}")
                            output_queue.put({
                                "type": "switch_result",
                                "success": False,
                                "message": f"Error: {str(e)}"
                            })
                    else:
                        output_queue.put({
                            "type": "switch_result",
                            "success": False,
                            "message": "Usage: /switch <provider> <model>"
                        })
                    
                    input_queue.task_done()
                    continue
                
                elif user_message == "/refresh":
                    # Handle refresh command
                    status_queue.put("Refreshing models...")
                    
                    refresh_tasks = []
                    for provider_name in chat_session.llm_client.PROVIDER_CONFIGS.keys():
                        if provider_name == "ollama" or chat_session.llm_client.config.get_api_key(provider_name) is not None:
                            task = asyncio.create_task(
                                chat_session.llm_client._fetch_provider_models(provider_name)
                            )
                            refresh_tasks.append(task)
                    
                    if refresh_tasks:
                        await asyncio.gather(*refresh_tasks, return_exceptions=True)
                    
                    # Update global variables
                    global provider_models, provider_health
                    provider_models = chat_session.llm_client.available_models
                    
                    # Update provider health
                    for provider, status in chat_session.llm_client.provider_health.items():
                        provider_health[provider] = "🟢 Healthy" if status else "🔴 Unhealthy"
                    
                    output_queue.put({
                        "type": "refresh_result",
                        "message": "Model refresh complete",
                        "providers": provider_models,
                        "health": provider_health
                    })
                    
                    input_queue.task_done()
                    continue
                
                # Regular message processing
                status_queue.put("Processing message...")
                
                # Add user message to history
                messages.append({"role": "user", "content": user_message})
                global chat_history
                chat_history.append({"role": "user", "content": user_message})
                
                # Get LLM response
                start_time = time.time()
                
                try:
                    llm_response, tokens = await chat_session.llm_client.get_response(messages)
                    
                    # Update token usage data
                    global token_usage
                    token_usage = {
                        "prompt_tokens": tokens.prompt_tokens,
                        "completion_tokens": tokens.completion_tokens,
                        "total_tokens": tokens.total_tokens
                    }
                    
                    # Calculate time taken
                    time_taken = time.time() - start_time
                    stats_info = f"\n[Model: {chat_session.llm_client.provider}/{chat_session.llm_client.model}] [Tokens: {tokens.prompt_tokens} in, {tokens.completion_tokens} out, {tokens.total_tokens} total] [Time: {time_taken:.2f}s]"
                    
                    # Look for tool calls in the response
                    tool_result = None
                    try:
                        import re
                        import json
                        
                        # Check if response is JSON (either direct or in code block)
                        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
                        match = re.search(json_pattern, llm_response)
                        
                        parsed_response = None
                        if match:
                            # Extract JSON from markdown
                            json_str = match.group(1)
                            try:
                                parsed_response = json.loads(json_str)
                            except json.JSONDecodeError:
                                pass
                        else:
                            # Try to parse as direct JSON
                            try:
                                parsed_response = json.loads(llm_response)
                            except json.JSONDecodeError:
                                pass
                            
                        # Handle tool execution if found
                        if parsed_response and "tool" in parsed_response and "arguments" in parsed_response:
                            tool_name = parsed_response['tool']
                            tool_arguments = parsed_response['arguments']
                            
                            status_queue.put(f"Executing tool: {tool_name}...")
                            logging.info(f"Executing tool: {tool_name}")
                            logging.info(f"With arguments: {tool_arguments}")
                            
                            # Execute the tool
                            found_server = False
                            for server in chat_session.servers:
                                tools = await server.list_tools()
                                if any(tool.name == tool_name for tool in tools):
                                    found_server = True
                                    try:
                                        tool_result = await server.execute_tool(tool_name, tool_arguments)
                                        
                                        # Handle the result
                                        serialized_result = chat_session.llm_client.safe_json_serialize(tool_result)
                                        tool_info = f"Tool execution result: {serialized_result}"
                                        
                                        # Add the original response to history
                                        messages.append({"role": "assistant", "content": llm_response})
                                        
                                        # Add the tool result to history as a system message
                                        messages.append({"role": "system", "content": tool_info})
                                        
                                        # Get a final response that interprets the tool result
                                        status_queue.put("Processing tool results...")
                                        start_time = time.time()
                                        final_response, final_tokens = await chat_session.llm_client.get_response(messages)
                                        time_taken = time.time() - start_time
                                        
                                        # Update token usage with the additional call
                                        token_usage["prompt_tokens"] += final_tokens.prompt_tokens
                                        token_usage["completion_tokens"] += final_tokens.completion_tokens
                                        token_usage["total_tokens"] += final_tokens.total_tokens
                                        
                                        final_stats = f"\n[Model: {chat_session.llm_client.provider}/{chat_session.llm_client.model}] [Tokens: {token_usage['prompt_tokens']} in, {token_usage['completion_tokens']} out, {token_usage['total_tokens']} total] [Time: {time_taken:.2f}s]"
                                        
                                        # Add the final response to history
                                        messages.append({"role": "assistant", "content": final_response})
                                        chat_history.append({"role": "assistant", "content": final_response + final_stats})
                                        
                                        # Return the response with tool result
                                        output_queue.put({
                                            "type": "chat_response",
                                            "message": final_response,
                                            "stats": final_stats,
                                            "tool_result": tool_result
                                        })
                                        break
                                    except Exception as e:
                                        error_msg = f"Error executing tool: {str(e)}"
                                        logging.error(error_msg)
                                        output_queue.put({
                                            "type": "chat_response",
                                            "message": f"Error executing tool: {tool_name}. {str(e)}",
                                            "stats": stats_info,
                                            "tool_result": None
                                        })
                                        chat_history.append({"role": "assistant", "content": f"Error executing tool: {tool_name}. {str(e)}" + stats_info})
                                        break
                            
                            if not found_server:
                                output_queue.put({
                                    "type": "chat_response",
                                    "message": f"No server found with tool: {tool_name}",
                                    "stats": stats_info,
                                    "tool_result": None
                                })
                                messages.append({"role": "assistant", "content": llm_response})
                                chat_history.append({"role": "assistant", "content": f"No server found with tool: {tool_name}" + stats_info})
                    except Exception as e:
                        logging.error(f"Error processing potential tool call: {e}")
                    
                    # If no tool was executed, return the direct response
                    if tool_result is None:
                        messages.append({"role": "assistant", "content": llm_response})
                        chat_history.append({"role": "assistant", "content": llm_response + stats_info})
                        output_queue.put({
                            "type": "chat_response",
                            "message": llm_response,
                            "stats": stats_info,
                            "tool_result": None
                        })
                    
                    status_queue.put("")  # Clear status
                except Exception as e:
                    error_message = f"Error getting LLM response: {str(e)}"
                    logging.error(error_message)
                    output_queue.put({
                        "type": "chat_response",
                        "message": f"An error occurred while processing your request: {str(e)}",
                        "stats": "",
                        "tool_result": None
                    })
                    chat_history.append({"role": "assistant", "content": f"An error occurred while processing your request: {str(e)}"})
                    status_queue.put("")  # Clear status
                
                input_queue.task_done()
            except queue.Empty:
                # No message in queue, just continue
                await asyncio.sleep(0.1)
                continue
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error(f"Error in message processor: {e}")
            status_queue.put(f"Error: {str(e)}")
            await asyncio.sleep(0.1)

def get_current_status():
    """Get the current status from the queue without blocking."""
    try:
        return status_queue.get(block=False)
    except queue.Empty:
        return None

def check_for_response():
    """Check for a response from the async thread."""
    try:
        result = output_queue.get(block=False)
        output_queue.task_done()
        
        if result["type"] == "chat_response":
            # Return the response with any tool result
            tool_output = None
            if result["tool_result"] is not None:
                if isinstance(result["tool_result"], dict):
                    tool_output = json.dumps(result["tool_result"], indent=2)
                else:
                    tool_output = str(result["tool_result"])
                
                # Truncate if very long
                if len(tool_output) > 1000:
                    tool_output = tool_output[:1000] + "...\n(output truncated)"
            
            return True, result["message"] + result["stats"], tool_output
        
        # For other types of responses, no chat update
        return False, None, None
    except queue.Empty:
        return False, None, None

def start_async_thread():
    """Start the async processing thread."""
    thread = threading.Thread(target=async_init)
    thread.daemon = True
    thread.start()
    return thread

# Start the async thread when the module loads
async_thread = start_async_thread()

# Routes
@app.route('/')
def index():
    """Render the main page."""
    # Always render the base template - initialization state is checked via API
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

# API routes
@app.route('/api/init-status')
def init_status():
    """Get the initialization status."""
    status = get_current_status() or "Initializing..."
    return jsonify({
        "status": status, 
        "is_initialized": is_initialized,
        "current_provider": current_provider or "Not set yet",
        "current_model": current_model or "Not set yet",
        "tools_count": len(available_tools),
        "resources_count": len(available_resources),
        "prompts_count": len(available_prompts),
        "provider_health": {k: str(v) for k, v in provider_health.items()} if provider_health else {}
    })

@app.route('/api/send-message', methods=['POST'])
def send_message():
    """Send a message to the LLM for processing."""
    try:
        data = request.json
        if 'message' not in data:
            return jsonify({"status": "error", "message": "No message provided"}), 400
        
        input_queue.put(data['message'])
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/check-response')
def api_check_response():
    """Check for a response from the LLM."""
    has_response, message, tool_result = check_for_response()
    return jsonify({
        "has_response": has_response,
        "message": message,
        "tool_result": tool_result,
        "token_usage": token_usage
    })

@app.route('/api/get-status')
def get_status():
    """Get the current status of the system."""
    status = get_current_status() or "Ready"
    return jsonify({
        "status": status,
        "token_usage": token_usage,
        "current_provider": current_provider,
        "current_model": current_model
    })

@app.route('/api/get-providers')
def get_providers():
    """Get the list of available providers."""
    provider_list = []
    for p, h in provider_health.items():
        provider_list.append(f"{p} {h}")
    return jsonify({"providers": provider_list})

@app.route('/api/get-models')
def get_models():
    """Get the models for a specific provider."""
    provider = request.args.get('provider', '')
    if not provider:
        return jsonify({"models": []})
    
    provider_name = provider.split()[0].lower()
    models = provider_models.get(provider_name, [])
    
    return jsonify({"models": models})

@app.route('/api/switch', methods=['POST'])
def switch_provider():
    """Switch the LLM provider and model."""
    try:
        data = request.json
        
        if 'provider' not in data or 'model' not in data:
            return jsonify({
                "status": "error",
                "message": "Provider and model are required"
            }), 400
        
        provider = data['provider']
        model = data['model']
        
        # Extract provider name
        provider_name = provider.split()[0].lower()
        
        # Submit command to async thread
        input_queue.put(f"/switch {provider_name} {model}")
        
        # Wait for result
        try:
            result = output_queue.get(timeout=30)
            output_queue.task_done()
            
            if result["type"] == "switch_result":
                return jsonify({
                    "status": "success" if result["success"] else "error",
                    "message": result["message"],
                    "current_provider": current_provider,
                    "current_model": current_model
                })
            
            return jsonify({
                "status": "error",
                "message": "Unexpected response type"
            })
        except queue.Empty:
            return jsonify({
                "status": "error",
                "message": "Timeout waiting for response"
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/refresh-models')
def refresh_all_models():
    """Refresh all LLM models."""
    try:
        input_queue.put("/refresh")
        
        try:
            result = output_queue.get(timeout=60)
            output_queue.task_done()
            
            if result["type"] == "refresh_result":
                return jsonify({
                    "status": "success",
                    "message": result["message"]
                })
            
            return jsonify({
                "status": "error",
                "message": "Unexpected response type"
            })
        except queue.Empty:
            return jsonify({
                "status": "error",
                "message": "Timeout waiting for response"
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/get-metadata')
def get_metadata():
    """Get metadata about tools, resources, and prompts."""
    try:
        metadata_html = generate_metadata_html()
        return jsonify({"content": metadata_html})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/force-init')
def force_init():
    """Force the initialization flag to true."""
    global is_initialized
    is_initialized = True
    return jsonify({"status": "success", "message": "Initialization forced"})

def main():
    try:
        # Configure the shutdown handler
        import atexit
        atexit.register(lambda: shutdown_event.set())
        
        # Start the Flask app
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
    finally:
        # Signal shutdown
        shutdown_event.set()

if __name__ == "__main__":
    # Configure asyncio policy for Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    main()