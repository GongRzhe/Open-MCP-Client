from flask import Flask, request, jsonify, render_template, send_from_directory
import asyncio
import json
import logging
import os
from threading import Thread
import sys

# Import from your existing chatbot module
from ChatMCP import (
    Configuration, Server, LLMClient, ChatSession,ConfigurationError
)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='web_chatbot.log',
    filemode='a'
)

# Global variables
chat_sessions = {}
config = None
servers = []
llm_client = None
server_config = None

def init_backend():
    """Initialize backend components"""
    global config, servers, llm_client, server_config
    
    config = Configuration()
    
    # Load server configuration
    try:
        server_config_dict = config.load_config('servers_config.json')
        
        # Create servers with the global config
        servers = [Server(name, srv_config, config) 
                   for name, srv_config in server_config_dict["mcpServers"].items()]
        
        # Create LLM client
        llm_client = LLMClient(config)
        
        # Initialize LLM client and servers in the background
        loop = asyncio.new_event_loop()
        
        async def initialize_components():
            # Initialize LLM client
            await llm_client.initialize()
            
            # Initialize servers
            init_tasks = []
            for server in servers:
                init_tasks.append(server.initialize())
            await asyncio.gather(*init_tasks, return_exceptions=True)
            
            logging.info("Backend components initialized")
        
        loop.run_until_complete(initialize_components())
        
        return True
    except Exception as e:
        logging.error(f"Error initializing backend: {e}")
        return False

# Initialize backend on startup
if not init_backend():
    logging.error("Failed to initialize backend. Exiting.")
    sys.exit(1)

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new chat session"""
    session_id = request.json.get('session_id', 'default')
    
    if session_id in chat_sessions:
        return jsonify({'status': 'exists', 'session_id': session_id})
    
    # Create new chat session object
    chat_session = ChatSession(servers, llm_client)
    chat_sessions[session_id] = {
        'session': chat_session,
        'messages': [],
        'initialized': False
    }
    
    return jsonify({'status': 'created', 'session_id': session_id})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a chat message"""
    data = request.json
    session_id = data.get('session_id', 'default')
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    if session_id not in chat_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = chat_sessions[session_id]
    chat_session = session_data['session']
    
    # Handle special commands
    if message.startswith('/'):
        result = handle_command(message, session_id)
        return jsonify(result)
    
    # Process message through chat session
    loop = asyncio.new_event_loop()
    
    try:
        # Define the async function
        async def process_message():
            if not session_data['initialized']:
                # Initialize the session if this is the first message
                all_tools = []
                for server in servers:
                    tools = await server.list_tools()
                    all_tools.extend(tools)
                
                # Add system message
                tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
                system_message = f"""You are a helpful assistant with access to these tools: 

{tools_description}
Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.

When you need to use a tool, respond with a valid JSON object in this format:
{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value"
    }}
}}
"""
                session_data['messages'].append({
                    "role": "system",
                    "content": system_message
                })
                session_data['initialized'] = True
            
            # Add user message
            session_data['messages'].append({
                "role": "user",
                "content": message
            })
            try:
                # Get LLM response
                llm_response, token_usage = await llm_client.get_response(session_data['messages'])
                
                # Convert TokenUsage object to a dictionary for JSON serialization
                token_info = {
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": token_usage.completion_tokens,
                    "total_tokens": token_usage.total_tokens
                }
                
                # Process response and execute tools if needed
                result = await chat_session.process_llm_response(llm_response)
                
                # If we got a result from a tool, get a final response
                if result != llm_response:
                    session_data['messages'].append({
                        "role": "assistant", 
                        "content": llm_response
                    })
                    session_data['messages'].append({
                        "role": "system",
                        "content": result
                    })
                    
                    # Get final response
                    final_response, final_token_usage = await llm_client.get_response(session_data['messages'])
                    
                    # Convert final token usage to dictionary
                    final_token_info = {
                        "prompt_tokens": final_token_usage.prompt_tokens,
                        "completion_tokens": final_token_usage.completion_tokens,
                        "total_tokens": final_token_usage.total_tokens
                    }
                    
                    session_data['messages'].append({
                        "role": "assistant",
                        "content": final_response
                    })
                    
                    return {
                        'response': final_response,
                        'tool_result': result,
                        'initial_response': llm_response,
                        'token_usage': token_info,
                        'final_token_usage': final_token_info,
                        'tool_used': True
                    }
                else:
                    # Add assistant response to history
                    session_data['messages'].append({
                        "role": "assistant",
                        "content": llm_response
                    })
                    
                    return {
                        'response': llm_response,
                        'token_usage': token_info,
                        'tool_used': False
                    }
            except Exception as e:
                logging.error(f"Error getting LLM response: {e}")
                return {
                    'error': f"Error connecting to the language model service: {str(e)}",
                    'response': "I encountered an error connecting to the language model service. Please try again in a moment."
                }
        # Run the async function and get the response
        response_data = loop.run_until_complete(process_message())
        
        # Return the jsonified response
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        return jsonify({'error': f'Error processing message: {str(e)}'}), 500
    finally:
        loop.close()

def handle_command(command, session_id):
    """Handle special commands"""
    cmd_parts = command.split()
    cmd = cmd_parts[0].lower()
    
    if cmd == '/llm':
        # Get available LLM providers and models
        providers = {}
        for provider_name, provider_info in llm_client.PROVIDER_CONFIGS.items():
            api_key_status = True if provider_name == "ollama" or llm_client.config.get_api_key(provider_name) else False
            
            models = llm_client.available_models.get(provider_name, [])
            if not models:
                models = provider_info["default_models"]
                
            providers[provider_name] = {
                'has_api_key': api_key_status,
                'models': models,
                'is_current': provider_name == llm_client.provider
            }
            
        return {
            'command_result': True,
            'command': 'llm',
            'providers': providers,
            'current_provider': llm_client.provider,
            'current_model': llm_client.model
        }
    
    elif cmd == '/switch' and len(cmd_parts) >= 3:
        provider = cmd_parts[1].lower()
        model = cmd_parts[2]
        
        # Switch provider and model
        loop = asyncio.new_event_loop()
        
        try:
            async def switch_provider():
                await llm_client.change_provider(provider, model)
                return {
                    'command_result': True,
                    'command': 'switch',
                    'provider': llm_client.provider,
                    'model': llm_client.model,
                    'message': f"Switched to {llm_client.provider.upper()} with model {llm_client.model}"
                }
                
            result = loop.run_until_complete(switch_provider())
            return result
        except ValueError as e:
            return {
                'command_result': False,
                'command': 'switch',
                'error': str(e)
            }
    
    elif cmd == '/refresh':
        # Refresh model lists
        loop = asyncio.new_event_loop()
        
        async def refresh_models():
            refresh_tasks = []
            for provider_name in llm_client.PROVIDER_CONFIGS.keys():
                if provider_name == "ollama" or llm_client.config.get_api_key(provider_name):
                    task = llm_client._fetch_provider_models(provider_name)
                    refresh_tasks.append(task)
            
            if refresh_tasks:
                await asyncio.gather(*refresh_tasks, return_exceptions=True)
            
            return {
                'command_result': True,
                'command': 'refresh',
                'message': "Model lists refreshed"
            }
            
        result = loop.run_until_complete(refresh_models())
        return result
    
    elif cmd == '/help':
        return {
            'command_result': True,
            'command': 'help',
            'commands': {
                '/llm': 'Show available LLM providers and models',
                '/switch <provider> <model>': 'Switch to a different LLM',
                '/refresh': 'Refresh model lists for all providers',
                '/help': 'Show this help message'
            }
        }
    
    return {
        'command_result': False,
        'command': cmd,
        'error': f"Unknown command: {cmd}"
    }

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status information"""
    provider_health = {}
    for provider, status in llm_client.provider_health.items():
        provider_health[provider] = {
            'healthy': status,
            'is_current': provider == llm_client.provider
        }
    
    server_status = {}
    for server in servers:
        server_status[server.name] = {
            'state': server.state.value,
            'tool_count': len(server.tools)
        }
    
    return jsonify({
        'llm': {
            'provider': llm_client.provider,
            'model': llm_client.model,
            'provider_health': provider_health
        },
        'servers': server_status,
        'sessions': list(chat_sessions.keys())
    })

@app.route('/api/tools', methods=['GET'])
def get_tools():
    """Get list of available tools"""
    loop = asyncio.new_event_loop()
    
    async def fetch_tools():
        all_tools = []
        for server in servers:
            tools = await server.list_tools()
            for tool in tools:
                all_tools.append({
                    'name': tool.name,
                    'description': tool.description,
                    'server': server.name
                })
        return all_tools
    
    tools = loop.run_until_complete(fetch_tools())
    return jsonify({'tools': tools})

if __name__ == '__main__':
    app.run(debug=True, port=5000)