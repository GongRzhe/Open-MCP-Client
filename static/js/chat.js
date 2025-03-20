/**
 * Chat functionality for ChatMCP Web Interface
 */

// Add a message to the chat display
function addMessage(content, isUser) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(isUser ? 'user-message' : 'assistant-message');
    messageDiv.innerHTML = `<p><strong>${isUser ? 'You' : 'Assistant'}:</strong> ${content}</p>`;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Submit a message to the server
function submitMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    
    if (message) {
        showLoading();
        addMessage(message, true);
        input.value = '';
        
        // Hide any previous tool results
        toggleToolResult(false);
        
        fetch('/api/send-message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Start checking for a response after a short delay
                setTimeout(checkForResponse, 1000);
            } else {
                console.error('Error sending message:', data.message);
                hideLoading();
            }
        })
        .catch(error => {
            console.error('Network error:', error);
            hideLoading();
            
            // Show error in chat
            addMessage(`Error: ${error.message}. Please try again.`, false);
        });
    }
}

// Process special commands
function processCommand(command) {
    // Strip the leading slash
    command = command.substring(1);
    
    const parts = command.split(' ');
    const cmd = parts[0].toLowerCase();
    
    switch (cmd) {
        case 'switch':
            if (parts.length >= 3) {
                const provider = parts[1];
                const model = parts.slice(2).join(' ');
                switchProvider(provider, model);
                return true;
            } else {
                addMessage("Usage: /switch [provider] [model]", false);
                return true;
            }
            
        case 'clear':
            clearChat();
            return true;
            
        case 'refresh':
            refreshModels();
            return true;
            
        case 'help':
            showHelp();
            return true;
            
        default:
            return false;
    }
}

// Clear the chat history
function clearChat() {
    const messagesDiv = document.getElementById('chat-messages');
    messagesDiv.innerHTML = '';
    addMessage("Chat history cleared.", false);
}

// Show help information
function showHelp() {
    const helpText = `
        <strong>Available Commands:</strong><br>
        /switch [provider] [model] - Switch to a different LLM<br>
        /clear - Clear chat history<br>
        /refresh - Refresh available models<br>
        /help - Show this help message
    `;
    addMessage(helpText, false);
}

// Switch to a different provider and model
function switchProvider(provider, model) {
    showLoading();
    
    fetch('/api/switch', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ provider, model })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            addMessage(`Switched to ${data.current_provider.toUpperCase()} with model ${data.current_model}`, false);
            updateProviderModel(data.current_provider, data.current_model);
        } else {
            addMessage(`Error: ${data.message}`, false);
        }
    })
    .catch(error => {
        addMessage(`Error: ${error.message}`, false);
    })
    .finally(() => {
        hideLoading();
    });
}

// Load chat history from server if available
function loadChatHistory() {
    fetch('/api/get-chat-history')
        .then(response => response.json())
        .then(data => {
            if (data.history && data.history.length > 0) {
                // Clear existing messages
                document.getElementById('chat-messages').innerHTML = '';
                
                // Add each message to the chat
                data.history.forEach(item => {
                    addMessage(item.content, item.role === 'user');
                });
            }
        })
        .catch(error => {
            console.error('Error loading chat history:', error);
        });
}

// Additional initialization for the chat interface
document.addEventListener('DOMContentLoaded', function() {
    // Try to load chat history if available
    if (typeof loadChatHistory === 'function') {
        try {
            loadChatHistory();
        } catch (e) {
            console.log('Chat history not available:', e);
        }
    }
    
    // Add support for command processing
    const userInput = document.getElementById('user-input');
    userInput.addEventListener('input', function() {
        // Simple command highlighting
        if (this.value.startsWith('/')) {
            this.style.color = '#0066cc';
        } else {
            this.style.color = '';
        }
    });
});