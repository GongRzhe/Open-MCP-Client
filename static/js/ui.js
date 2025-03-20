/**
 * UI Interaction Functions for ChatMCP Web Interface
 */

// Toggle accordion sections open/closed
function toggleAccordion(id) {
    const content = document.getElementById(id);
    content.classList.toggle('visible');
}

// Show the loading indicator
function showLoading() {
    document.getElementById('loading').classList.add('active');
}

// Hide the loading indicator
function hideLoading() {
    document.getElementById('loading').classList.remove('active');
}

// Show or hide the tool result panel
function toggleToolResult(show, content = null) {
    const toolResult = document.getElementById('tool-result');
    const toolOutput = document.getElementById('tool-output');
    
    if (show && content) {
        toolOutput.textContent = content;
        toolResult.style.display = 'block';
    } else {
        toolResult.style.display = 'none';
    }
}

// Update the displayed status 
function updateStatus(status) {
    document.getElementById('tool-status').textContent = 'Status: ' + status;
}

// Update the displayed token usage
function updateTokenUsage(usage) {
    document.getElementById('token-usage').textContent = 
        JSON.stringify(usage, null, 2);
}

// Update the current provider and model display
function updateProviderModel(provider, model) {
    document.getElementById('current-provider-model').innerHTML = 
        `Provider: ${provider.toUpperCase()}<br>Model: ${model}`;
}

// Initialize UI components when DOM loads
document.addEventListener('DOMContentLoaded', function() {
    // Open first accordion by default
    document.getElementById('current-llm').classList.add('visible');
    
    // Add event listeners
    document.getElementById('refresh-status').addEventListener('click', refreshStatus);
    document.getElementById('refresh-metadata-btn').addEventListener('click', refreshMetadata);
    
    // LLM switching
    document.getElementById('fetch-providers-btn').addEventListener('click', fetchProviders);
    document.getElementById('fetch-models-btn').addEventListener('click', fetchModels);
    document.getElementById('switch-btn').addEventListener('click', switchProviderModel);
    document.getElementById('refresh-btn').addEventListener('click', refreshModels);
    
    // Initialize chat area
    initializeChatArea();
    
    // Do an initial status refresh
    refreshStatus();
});

// Initialize chat area 
function initializeChatArea() {
    document.getElementById('send-btn').addEventListener('click', submitMessage);
    document.getElementById('user-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            submitMessage();
        }
    });
    document.getElementById('check-response').addEventListener('click', checkForResponse);
}

// Check initialization status
function checkInitStatus() {
    fetch('/api/init-status')
        .then(response => response.json())
        .then(data => {
            let statusText = `Status: ${data.status}\n`;
            statusText += `Initialized: ${data.is_initialized}\n`;
            statusText += `Provider: ${data.current_provider}\n`;
            statusText += `Model: ${data.current_model}\n`;
            statusText += `Tools: ${data.tools_count}, Resources: ${data.resources_count}, Prompts: ${data.prompts_count}\n`;
            
            if (Object.keys(data.provider_health).length > 0) {
                statusText += "Provider Health:\n";
                for (const [provider, health] of Object.entries(data.provider_health)) {
                    statusText += `- ${provider}: ${health}\n`;
                }
            }
            
            if (document.getElementById('init-status')) {
                document.getElementById('init-status').innerHTML = statusText.replace(/\n/g, '<br>');
                
                if (data.is_initialized && document.getElementById('show-main-ui')) {
                    document.getElementById('show-main-ui').style.display = 'inline-block';
                }
            }
        })
        .catch(error => {
            console.error('Error checking initialization status:', error);
        });
}

// Start auto-refresh functionality
function setupAutoRefresh() {
    // Variables to control refresh rates
    const STATUS_REFRESH_INTERVAL = 3000;  // 3 seconds
    const RESPONSE_CHECK_INTERVAL = 2000;  // 2 seconds
    
    // Flag to track if we're waiting for a response
    let waitingForResponse = false;
    
    // Function to check for pending responses
    function autoCheckResponse() {
        // Only check if we're waiting for a response
        if (waitingForResponse) {
            console.log('Auto-checking for response...');
            checkForResponse();
        }
    }
    
    // Function to refresh status
    function autoRefreshStatus() {
        console.log('Auto-refreshing status...');
        refreshStatus();
    }
    
    // Set up the intervals
    setInterval(autoRefreshStatus, STATUS_REFRESH_INTERVAL);
    setInterval(autoCheckResponse, RESPONSE_CHECK_INTERVAL);
    
    // Override the send button to set the waiting flag
    const originalSubmitMessage = window.submitMessage;
    window.submitMessage = function() {
        // Call the original function
        originalSubmitMessage();
        
        // Set the flag to start checking for responses
        waitingForResponse = true;
        
        // Schedule a check right away
        setTimeout(autoCheckResponse, 500);
    };
    
    // Override checkForResponse to clear the flag when a response is received
    const originalCheckForResponse = window.checkForResponse;
    window.checkForResponse = function() {
        fetch('/api/check-response')
            .then(response => response.json())
            .then(data => {
                if (data.has_response) {
                    // Response received, clear the waiting flag
                    waitingForResponse = false;
                    
                    // Add message to chat
                    addMessage(data.message, false);
                    
                    // Show tool result if available
                    if (data.tool_result) {
                        toggleToolResult(true, data.tool_result);
                    }
                    
                    // Update token usage
                    updateTokenUsage(data.token_usage);
                }
            })
            .finally(() => {
                hideLoading();
            });
    };
    
    console.log('Auto-refresh functionality initialized');
}

// Wait for document to be ready before setting up auto-refresh
document.addEventListener('DOMContentLoaded', function() {
    // Setup auto-refresh with a small delay to ensure everything is loaded
    setTimeout(setupAutoRefresh, 500);
});