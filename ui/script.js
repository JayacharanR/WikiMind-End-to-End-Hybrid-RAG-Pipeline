const API_URL = "http://localhost:8000";

// DOM Elements
const chatHistory = document.getElementById('chatHistory');
const emptyState = document.getElementById('emptyState');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');

// Dropdown Elements
const settingsBtn = document.getElementById('settingsBtn');
const settingsDropdown = document.getElementById('settingsDropdown');
const dropdownArrow = document.getElementById('dropdownArrow');
const toggleTimeTravel = document.getElementById('toggleTimeTravel');
const datePickerContainer = document.getElementById('datePickerContainer');

// Dropdown Logic
settingsBtn.addEventListener('click', (e) => {
    e.preventDefault();
    settingsDropdown.classList.toggle('show');
    dropdownArrow.classList.toggle('rotated');
});

// Sidebar Toggle Logic

// Settings State
const strategies = {
    multi_query: false,
    hyde: false,
    step_back: false,
    decomposition: false,
    as_of_date: null
};

// Sidebar Toggle Logic
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const chatContainer = document.querySelector('.chat-container');

sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    chatContainer.classList.toggle('expanded');
});

// Update settings state on change
document.getElementById('toggleMultiQuery').addEventListener('change', (e) => strategies.multi_query = e.target.checked);
document.getElementById('toggleHyde').addEventListener('change', (e) => strategies.hyde = e.target.checked);
document.getElementById('toggleStepBack').addEventListener('change', (e) => strategies.step_back = e.target.checked);
document.getElementById('toggleDecomposition').addEventListener('change', (e) => strategies.decomposition = e.target.checked);

toggleTimeTravel.addEventListener('change', (e) => {
    if (e.target.checked) {
        datePickerContainer.style.display = 'block';
        updateDate();
    } else {
        datePickerContainer.style.display = 'none';
        strategies.as_of_date = null;
    }
});

const asOfDateInput = document.getElementById('asOfDate');
asOfDateInput.valueAsDate = new Date();
asOfDateInput.addEventListener('change', updateDate);

function updateDate() {
    if (toggleTimeTravel.checked && asOfDateInput.value) {
        // format to ISO with T23:59:59Z
        strategies.as_of_date = asOfDateInput.value + "T23:59:59Z";
    }
}


// Auto-resize textarea
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Remove empty state
    if (emptyState) {
        emptyState.style.display = 'none';
    }

    // Add User Message
    appendMessage('user', text);
    chatInput.value = '';
    chatInput.style.height = 'auto'; // reset height

    // Setup Assistant Message placeholder
    const { bubble, contentDiv, metaDiv } = createMessageContainer('assistant');
    
    // Prepare Payload
    const payload = {
        query: text,
        strategies: {
            multi_query: strategies.multi_query,
            hyde: strategies.hyde,
            step_back: strategies.step_back,
            decomposition: strategies.decomposition
        }
    };
    if (strategies.as_of_date) {
        payload.as_of_date = strategies.as_of_date;
    }

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            contentDiv.innerHTML = `<span style="color:red">Error: Backend returned ${response.status}</span>`;
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        
        let fullAnswer = "";
        let isDone = false;
        let currentEvent = null;

        contentDiv.innerHTML = '<em>Agent is working...</em>';

        while (!isDone) {
            const { value, done } = await reader.read();
            if (done) {
                isDone = true;
                break;
            }
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    currentEvent = line.substring(7).trim();
                }
                else if (line.startsWith('data: ')) {
                    const dataStr = line.substring(6).trim();
                    if (!dataStr) continue;
                    
                    try {
                        const data = JSON.parse(dataStr);
                        
                        if (currentEvent === "update") {
                            contentDiv.innerHTML = `<em>Agent is working... [${data.node}]</em>`;
                        } 
                        else if (currentEvent === "final") {
                            let answer = data.answer || "";
                            try {
                                const ansJson = JSON.parse(answer);
                                if (ansJson.name === "generate_text" && ansJson.parameters) {
                                    let params = ansJson.parameters;
                                    if (typeof params === 'string') params = JSON.parse(params);
                                    answer = params.input || answer;
                                }
                            } catch(e) {}
                            
                            fullAnswer = answer;
                            contentDiv.innerHTML = marked.parse(fullAnswer);
                            
                            // Render Sources / Metadata if present
                            let metaHtml = '';
                            if (data.metadata && data.metadata.total_duration_ms) {
                                const sec = (data.metadata.total_duration_ms / 1000).toFixed(2);
                                metaHtml += `<div class="metadata-badge"><i data-feather="clock" style="width:12px;height:12px"></i> ${sec}s</div>`;
                            }
                            if (data.sources && data.sources.length > 0) {
                                metaHtml += `<div class="metadata-badge"><i data-feather="book-open" style="width:12px;height:12px"></i> ${data.sources.length} sources</div>`;
                            }
                            metaDiv.innerHTML = metaHtml;
                            feather.replace(); // render new icons
                        }
                        else if (currentEvent === "error") {
                            contentDiv.innerHTML = `<span style="color:red">Backend Error: ${data.detail || 'Unknown error'}</span>`;
                        }
                    } catch (e) {
                        console.error("Error parsing JSON chunk", e, dataStr);
                    }
                }
            }
            scrollToBottom();
        }

    } catch (err) {
        console.error(err);
        contentDiv.innerHTML = `<span style="color:red">Failed to connect to backend: ${err.message}</span>`;
    }
}

function appendMessage(role, text) {
    const { bubble, contentDiv } = createMessageContainer(role);
    if (role === 'user') {
        contentDiv.textContent = text; // Plain text for user
    } else {
        contentDiv.innerHTML = marked.parse(text); // Markdown for assistant
    }
    scrollToBottom();
}

function createMessageContainer(role) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message ${role}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    bubble.appendChild(contentDiv);
    
    const metaDiv = document.createElement('div');
    metaDiv.className = 'metadata';
    bubble.appendChild(metaDiv);

    msgDiv.appendChild(bubble);
    chatHistory.appendChild(msgDiv);
    
    return { msgDiv, bubble, contentDiv, metaDiv };
}

function scrollToBottom() {
    chatHistory.scrollTop = chatHistory.scrollHeight;
}
