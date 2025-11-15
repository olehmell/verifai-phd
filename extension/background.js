// Background service worker for VerifAI extension

const DEFAULT_API_URL = 'http://localhost:8000/analyze'; // Use test endpoint by default

// Initialize context menu on extension install
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'verify-with-verifai',
    title: 'Verify with VerifAI',
    contexts: ['selection']
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === 'verify-with-verifai' && info.selectionText) {
    const selectedText = info.selectionText.trim();
    
    if (!selectedText) {
      return;
    }

    // Get API URL from storage or use default
    const result = await chrome.storage.sync.get(['apiUrl']);
    const apiUrl = result.apiUrl || DEFAULT_API_URL;

    // Always ensure content script is injected before sending messages
    try {
      // Try to ping the content script first
      await chrome.tabs.sendMessage(tab.id, { action: 'ping' });
    } catch (error) {
      // Content script not loaded, inject it
      console.log('Injecting content script...');
      try {
        await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files: ['content.js']
        });
        await chrome.scripting.insertCSS({
          target: { tabId: tab.id },
          files: ['popup.css']
        });
        // Wait for script to initialize
        await new Promise(resolve => setTimeout(resolve, 150));
      } catch (injectError) {
        console.error('Failed to inject content script:', injectError);
        return;
      }
    }

    // Send message to content script to show loading state
    try {
      await chrome.tabs.sendMessage(tab.id, {
        action: 'showLoading',
        selectedText: selectedText
      });
    } catch (error) {
      console.error('Failed to send loading message after injection:', error);
      return;
    }

    try {
      // Make API call
      console.log('[Background] Starting API call to:', apiUrl);
      console.log('[Background] Selected text length:', selectedText.length);
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          content: selectedText
        })
      });

      console.log('[Background] API response status:', response.status, response.statusText);

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('[Background] API response data received:', data);
      console.log('[Background] Data keys:', Object.keys(data));
      console.log('[Background] Manipulation:', data.manipulation);
      console.log('[Background] Techniques count:', data.techniques?.length || 0);

      // Send results to content script
      console.log('[Background] Attempting to send showResults message to tab:', tab.id);
      try {
        await chrome.tabs.sendMessage(tab.id, {
          action: 'showResults',
          data: data,
          selectedText: selectedText
        });
        console.log('[Background] showResults message sent successfully');
      } catch (msgError) {
        console.error('[Background] Failed to send results message:', msgError);
        console.error('[Background] Error details:', JSON.stringify(msgError));
        // Try to re-inject content script and retry
        try {
          console.log('[Background] Attempting to re-inject content script');
          await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            files: ['content.js']
          });
          await chrome.scripting.insertCSS({
            target: { tabId: tab.id },
            files: ['popup.css']
          });
          await new Promise(resolve => setTimeout(resolve, 150));
          console.log('[Background] Retrying showResults message after re-injection');
          await chrome.tabs.sendMessage(tab.id, {
            action: 'showResults',
            data: data,
            selectedText: selectedText
          });
          console.log('[Background] showResults message sent successfully after re-injection');
        } catch (retryError) {
          console.error('[Background] Failed to send results after re-injection:', retryError);
          console.error('[Background] Retry error details:', JSON.stringify(retryError));
        }
      }
    } catch (error) {
      console.error('[Background] VerifAI API error:', error);
      console.error('[Background] Error stack:', error.stack);
      
      // Send error to content script
      try {
        // Ensure content script is loaded
        try {
          await chrome.tabs.sendMessage(tab.id, { action: 'ping' });
        } catch (pingError) {
          // Inject if not loaded
          await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            files: ['content.js']
          });
          await chrome.scripting.insertCSS({
            target: { tabId: tab.id },
            files: ['popup.css']
          });
          await new Promise(resolve => setTimeout(resolve, 150));
        }
        
        await chrome.tabs.sendMessage(tab.id, {
          action: 'showError',
          error: error.message
        });
      } catch (msgError) {
        console.error('Failed to send error message:', msgError);
      }
    }
  }
});

// Handle messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'closePopup') {
    chrome.tabs.sendMessage(sender.tab.id, {
      action: 'closePopup'
    });
  }
  return true;
});

