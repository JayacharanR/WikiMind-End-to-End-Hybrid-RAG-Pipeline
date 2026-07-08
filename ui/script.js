const API_URL = "http://localhost:8000";

let isGenerating = false;
let currentAbortController = null;

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
const topNav = document.querySelector('.top-nav');

sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    chatContainer.classList.toggle('expanded');
    topNav.classList.toggle('collapsed');
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
        if (!isGenerating) {
            sendMessage();
        }
    }
});

sendBtn.addEventListener('click', () => {
    if (isGenerating && currentAbortController) {
        currentAbortController.abort();
    } else if (!isGenerating) {
        sendMessage();
    }
});

async function sendMessage() {
    if (isGenerating) return;

    const text = chatInput.value.trim();
    // Trigger slide down animation
    const inputContainer = document.getElementById('inputContainer');
    if (inputContainer && inputContainer.classList.contains('centered')) {
        inputContainer.classList.remove('centered');
    }

    // Hide empty state
    const emptyState = document.getElementById('emptyState');
    if (emptyState && emptyState.style.display !== 'none') {
        emptyState.style.opacity = '0';
        setTimeout(() => {
            emptyState.style.display = 'none';
        }, 400); // Wait for fade out
    }

    // Add User Message
    appendMessage('user', text);
    chatInput.value = '';
    chatInput.style.height = 'auto'; // reset height

    isGenerating = true;
    currentAbortController = new AbortController();
    
    // Change icon to stop
    sendBtn.innerHTML = '<i data-feather="square"></i>';
    if (window.feather) feather.replace();

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

    const startTime = performance.now();

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
            },
            body: JSON.stringify(payload),
            signal: currentAbortController.signal
        });

        if (!response.ok) {
            contentDiv.innerHTML = `<span style="color:red">Error: Backend returned ${response.status}</span>`;
            return;
        }

        const contentType = response.headers.get('content-type');
        
        // Handle Cache Hit (JSON Response)
        if (contentType && contentType.includes('application/json')) {
            const data = await response.json();
            renderFinalResponse(data, contentDiv, metaDiv, startTime);
            return;
        }

        // Handle Cache Miss (SSE Stream)
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        
        let isDone = false;
        let currentEvent = null;

        contentDiv.innerHTML = '<div class="loading-text"><i data-feather="loader" class="spin-icon"></i> <em>Agent is working...</em></div>';
        if (window.feather) feather.replace();

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
                            contentDiv.innerHTML = `<div class="loading-text"><i data-feather="loader" class="spin-icon"></i> <em>Agent is working... [${data.node}]</em></div>`;
                            if (window.feather) feather.replace();
                        } 
                        else if (currentEvent === "final") {
                            renderFinalResponse(data, contentDiv, metaDiv, startTime);
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
        if (err.name === 'AbortError') {
            contentDiv.innerHTML = '<em>Query stopped by user.</em>';
        } else {
            console.error(err);
            contentDiv.innerHTML = `<span style="color:red">Failed to connect to backend: ${err.message}</span>`;
        }
    } finally {
        isGenerating = false;
        currentAbortController = null;
        sendBtn.innerHTML = '<i data-feather="arrow-up"></i>';
        if (window.feather) feather.replace();
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

function renderFinalResponse(data, contentDiv, metaDiv, startTime) {
    let answer = data.answer || "";
    try {
        const ansJson = JSON.parse(answer);
        if (ansJson.name === "generate_text" && ansJson.parameters) {
            let params = ansJson.parameters;
            if (typeof params === 'string') params = JSON.parse(params);
            answer = params.input || answer;
        }
    } catch(e) {}
    
    let fullAnswer = answer || "";
    
    if (data.metadata && data.metadata.expanded_queries && data.metadata.expanded_queries.length > 0) {
        fullAnswer += "\n\n---\n\n**Queries Searched:**\n";
        data.metadata.expanded_queries.forEach(q => {
            fullAnswer += `- ${q}\n`;
        });
    }

    let uniqueUrls = new Set();
    
    if (data.sources && data.sources.length > 0) {
        fullAnswer += "\n\n---\n\n**References:**\n";
        let refCount = 0;
        data.sources.forEach(src => {
            let url = src.url;
            if (!url && src.title) {
                url = "https://en.wikipedia.org/wiki/" + encodeURIComponent(src.title.replace(/ /g, '_'));
            }
            
            if (url && !uniqueUrls.has(url)) {
                uniqueUrls.add(url);
                if (refCount < 5) { // Limit to maximum 5 articles
                    fullAnswer += `- [${src.title || url}](${url})\n`;
                    refCount++;
                }
            }
        });
    } else {
        fullAnswer += "\n\n---\n\n**References:**\n- *No relevant Wikipedia articles were found in the local database. The answer above was generated using the model's internal knowledge.*";
    }
    
    contentDiv.innerHTML = marked.parse(fullAnswer);
    
    // Render Sources / Metadata if present
    let metaHtml = '';
    
    // Add Timer
    let durationSec = ((performance.now() - startTime) / 1000).toFixed(2);
    metaHtml += `<div class="metadata-badge"><i data-feather="clock" style="width:12px;height:12px"></i> ${durationSec}s</div>`;
    
    if (uniqueUrls.size > 0) {
        metaHtml += `<div class="metadata-badge"><i data-feather="book-open" style="width:12px;height:12px"></i> ${uniqueUrls.size} articles</div>`;
    }
    
    // Add Copy Button
    metaHtml += `<button class="copy-btn metadata-badge" style="background:transparent; border:1px solid #333; color:#A0A0A0; cursor:pointer;" title="Copy to clipboard">
        <i data-feather="copy" style="width:12px;height:12px"></i> Copy
    </button>`;
    
    metaDiv.innerHTML = metaHtml;
    
    // Attach copy event listener
    const copyBtn = metaDiv.querySelector('.copy-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(fullAnswer).then(() => {
                copyBtn.innerHTML = `<i data-feather="check" style="width:12px;height:12px"></i> Copied`;
                if (window.feather) feather.replace();
                setTimeout(() => {
                    copyBtn.innerHTML = `<i data-feather="copy" style="width:12px;height:12px"></i> Copy`;
                    if (window.feather) feather.replace();
                }, 2000);
            });
        });
    }

    if (window.feather) {
        feather.replace();
    }
}
