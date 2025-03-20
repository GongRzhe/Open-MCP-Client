/**
 * API communication functions for ChatMCP Web Interface
 */

// Refresh status information
function refreshStatus() {
    fetch('/api/get-status')
        .then(response => response.json())
        .then(data => {
            updateStatus(data.status);
            updateTokenUsage(data.token_usage);
            updateProviderModel(data.current_provider, data.current_model);
        })
        .catch(error => {
            console.error('Error refreshing status:', error);
        });
}

// Fetch the list of available providers
function fetchProviders() {
    fetch('/api/get-providers')
        .then(response => response.json())
        .then(data => {
            const dropdown = document.getElementById('provider-dropdown');
            dropdown.innerHTML = '<option value="">Select a provider</option>';
            
            data.providers.forEach(provider => {
                const option = document.createElement('option');
                option.value = provider;
                option.textContent = provider;
                dropdown.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Error fetching providers:', error);
        });
}

// Fetch models for the selected provider
function fetchModels() {
    const provider = document.getElementById('provider-dropdown').value;
    if (!provider) return;
    
    fetch(`/api/get-models?provider=${encodeURIComponent(provider)}`)
        .then(response => response.json())
        .then(data => {
            const dropdown = document.getElementById('model-dropdown');
            dropdown.innerHTML = '<option value="">Select a model</option>';
            
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                dropdown.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Error fetching models:', error);
        });
}

// Switch the provider and model
function switchProviderModel() {
    const provider = document.getElementById('provider-dropdown').value;
    const model = document.getElementById('model-dropdown').value;
    
    if (!provider || !model) {
        document.getElementById('switch-status').textContent = 'Please select both provider and model';
        return;
    }
    
    showLoading();
    document.getElementById('switch-status').textContent = 'Switching...';
    
    fetch('/api/switch', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ provider, model })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('switch-status').textContent = data.message;
        updateProviderModel(data.current_provider, data.current_model);
    })
    .catch(error => {
        document.getElementById('switch-status').textContent = 'Error: ' + error.message;
        console.error('Error switching provider/model:', error);
    })
    .finally(() => {
        hideLoading();
    });
}

// Refresh all models for all providers
function refreshModels() {
    showLoading();
    document.getElementById('switch-status').textContent = 'Refreshing models...';
    
    fetch('/api/refresh-models')
        .then(response => response.json())
        .then(data => {
            document.getElementById('switch-status').textContent = data.message;
            fetchProviders();
        })
        .catch(error => {
            document.getElementById('switch-status').textContent = 'Error: ' + error.message;
            console.error('Error refreshing models:', error);
        })
        .finally(() => {
            hideLoading();
        });
}

// Refresh metadata (tools, resources, prompts)
function refreshMetadata() {
    fetch('/api/get-metadata')
        .then(response => response.json())
        .then(data => {
            document.getElementById('metadata-content').innerHTML = data.content;
        })
        .catch(error => {
            console.error('Error refreshing metadata:', error);
        });
}

// Check for a response from the server
function checkForResponse() {
    fetch('/api/check-response')
        .then(response => response.json())
        .then(data => {
            if (data.has_response) {
                // Update chat with response
                addMessage(data.message, false);
                
                // Show tool result if available
                if (data.tool_result) {
                    toggleToolResult(true, data.tool_result);
                }
                
                // Update token usage
                updateTokenUsage(data.token_usage);
            }
        })
        .catch(error => {
            console.error('Error checking for response:', error);
        })
        .finally(() => {
            hideLoading();
        });
}

// Force initialization (for cases where initialization gets stuck)
function forceInitialization() {
    fetch('/api/force-init')
        .then(response => response.json())
        .then(data => {
            console.log('Initialization forced:', data);
            // Redirect to main page
            window.location.href = '/';
        })
        .catch(error => {
            console.error('Error forcing initialization:', error);
        });
}