# ChatMCP - Multi-Provider LLM Chat Client with Tool Integration

ChatMCP is a powerful command-line chat interface that connects to multiple LLM providers (OpenAI, Anthropic, Groq, etc.) and extends their capabilities with tools using the Model Context Protocol (MCP).

## Features

- 🔄 **Multiple LLM Provider Support**: Seamlessly switch between OpenAI, Anthropic, Groq, Gemini, Ollama, and OpenRoute
- 🧰 **Integrated Tools**: Search the web, manage GitHub repositories, browse websites, and more through MCP servers
- 🔁 **Resilient Design**: Built with circuit breakers, automatic retries, health checks, and graceful degradation
- ⚡ **Asynchronous Architecture**: Efficiently handles multiple connections and operations concurrently
- 🔍 **Token Tracking**: Monitors token usage across different providers
- 📊 **Performance Metrics**: Reports response times and token counts for each interaction
- 📝 **Comprehensive Logging**: Detailed activity logging for debugging and monitoring

## Getting Started

### Prerequisites

- Python 3.8+
- API keys for the LLM providers you want to use

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/chatmcp.git
   cd chatmcp
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Copy the example servers_config file and add your mcp servers:
   ```bash
   cp servers_config.json.example servers_config.json
   ```

### Configuration

1. **Environment Variables**: Edit the `.env` file with your API keys and preferences:
   ```
   # LLM API Keys
   ANTHROPIC_API_KEY=your_anthropic_key
   OPENAI_API_KEY=your_openai_key
   OPENROUTE_API_KEY=your_openroute_key
   GEMINI_API_KEY=your_gemini_key
   GROQ_API_KEY=your_groq_key
   
   # Ollama settings (if using local Ollama)
   OLLAMA_HOST=http://localhost:11434
   
   # Default provider and model configuration
   DEFAULT_LLM_PROVIDER=groq
   DEFAULT_LLM_MODEL=llama-3.2-90b-vision-preview
   ```

2. **Server Configuration**: Review and modify `servers_config.json` to configure the MCP servers:
   ```json
   {
     "mcpServers": {
       "github": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-github"],
         "env": {
           "GITHUB_PERSONAL_ACCESS_TOKEN": "your_github_token"
         }
       },
       "brave-search": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-brave-search"],
         "env": {
           "BRAVE_API_KEY": "your_brave_api_key"
         },
         "disabled": false,
         "autoApprove": []
       }
       // ... other servers
     }
   }
   ```

## Usage

### Starting the Chat

Run the application with:

```bash
python ChatMCP.py
```

### Chat Commands

Once the chat is running, you can use these commands:

- `/llm` - Display available LLM providers and models
- `/switch <provider> <model>` - Switch to a different LLM provider and model
  - Example: `/switch openai gpt-4o`
- `/refresh` - Refresh the list of available models
- `/tools` - Show available MCP tools
- `/now` - Show current LLM details and token usage
- `/help` - Display help information
- `quit` or `exit` - Exit the chat

### Chatting with the Assistant

Simply type your message and press Enter. The assistant will respond, and if appropriate, it will use one of the available tools to provide enhanced information.

```
You: What's the weather in New York?
Assistant: I'll check the current weather in New York for you.

> Executing tool: brave_web_search
> With arguments: {"query": "current weather in New York"}
> Tool execution completed: brave_web_search

Based on the latest information, New York is currently experiencing temperatures around 72°F (22°C) with partly cloudy skies. Humidity is at 65% with light winds from the southwest at 5-10 mph. There's no precipitation expected in the next few hours.

[Model: openai/gpt-4o] [Tokens: 145 in, 98 out, 243 total] [Time: 1.45s]
```

## Architecture

ChatMCP is built with a robust, asyncio-based architecture for high performance and reliability:

- **Server Management**: Connects to multiple MCP servers concurrently
- **LLM Client**: Manages connections to various LLM providers
- **Chat Session**: Orchestrates the interaction between user, LLM, and tools
- **Circuit Breaker Pattern**: Prevents cascading failures when services are unavailable
- **Async Retry**: Implements exponential backoff with jitter for reliable connections

## Supported LLM Providers

- OpenAI (ChatGPT models)
- Anthropic (Claude models)
- Groq (Llama models)
- Google Gemini
- OpenRoute (proxy for various models)
- Ollama (local models)

## Available Tools

Depending on your configuration, ChatMCP can provide tools for:

- Web search (via Brave Search)
- GitHub integration (repository management, file operations)
- Web browsing and scraping (via Puppeteer)
- Gmail integration
- File fetching and processing
- Memory persistence

## Troubleshooting

### Connection Issues

If you encounter connection problems with LLM providers:

1. Check your API keys in the `.env` file
2. Verify your internet connection
3. See if the provider's service is operational
4. Check the logs in `chatbot.log`

### Server Startup Problems

If MCP servers fail to start:

1. Ensure you have Node.js installed (required for npx commands)
2. Check if the required NPM packages are accessible
3. Verify any required API keys are properly set in the config
4. Check for conflicts with other services on the same ports

## Advanced Configuration

### Modifying Timeouts and Retries

You can add these settings to your `.env` file:

```
CONNECTION_TIMEOUT=10
READ_TIMEOUT=60
MAX_RETRIES=3
RETRY_DELAY_BASE=1.0
RETRY_MAX_DELAY=30.0
```

### Message History Limit

Control conversation length to manage token usage:

```
MESSAGE_HISTORY_LIMIT=20
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the application.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
