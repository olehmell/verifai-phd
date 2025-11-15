// Content script for VerifAI extension

let popupElement = null;
let selectedTextRange = null;

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('[Content] Received message:', request.action);
  console.log('[Content] Request object:', request);
  
  // Handle ping immediately
  if (request.action === 'ping') {
    console.log('[Content] Responding to ping');
    sendResponse({ success: true, ready: true });
    return false; // Synchronous response
  }
  
  // Handle other actions
  try {
    if (request.action === 'showLoading') {
      console.log('[Content] Handling showLoading');
      showLoadingPopup(request.selectedText);
      sendResponse({ success: true });
    } else if (request.action === 'showResults') {
      console.log('[Content] Handling showResults');
      console.log('[Content] Results data:', request.data);
      console.log('[Content] Popup element exists:', !!popupElement);
      showResultsPopup(request.data, request.selectedText);
      sendResponse({ success: true });
    } else if (request.action === 'showError') {
      console.log('[Content] Handling showError');
      showErrorPopup(request.error);
      sendResponse({ success: true });
    } else if (request.action === 'closePopup') {
      console.log('[Content] Handling closePopup');
      closePopup();
      sendResponse({ success: true });
    } else {
      console.warn('[Content] Unknown action:', request.action);
      sendResponse({ success: false, error: 'Unknown action' });
    }
  } catch (error) {
    console.error('[Content] Error handling message:', error);
    console.error('[Content] Error stack:', error.stack);
    sendResponse({ success: false, error: error.message });
  }
  
  return true; // Keep the message channel open for async response
});

function getSelectionRange() {
  const selection = window.getSelection();
  if (selection.rangeCount > 0) {
    return selection.getRangeAt(0);
  }
  return null;
}

function getPopupPosition(range) {
  const rect = range.getBoundingClientRect();
  const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
  const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;

  // Position popup below the selection, aligned to the left
  let top = rect.bottom + scrollTop + 10;
  let left = rect.left + scrollLeft;

  // Ensure popup doesn't go off screen
  const popupWidth = 400;
  const popupHeight = 300;
  const windowWidth = window.innerWidth;
  const windowHeight = window.innerHeight;

  if (left + popupWidth > windowWidth) {
    left = windowWidth - popupWidth - 20;
  }
  if (left < 0) {
    left = 20;
  }
  if (top + popupHeight > windowHeight + scrollTop) {
    // Position above selection instead
    top = rect.top + scrollTop - popupHeight - 10;
  }
  if (top < scrollTop) {
    top = scrollTop + 20;
  }

  return { top, left };
}

function createPopup() {
  // Remove existing popup if any
  closePopup();

  // Create popup container
  popupElement = document.createElement('div');
  popupElement.id = 'verifai-popup';
  popupElement.className = 'verifai-popup';
  popupElement.style.display = 'block';
  popupElement.style.visibility = 'visible';
  popupElement.style.opacity = '1';
  popupElement.style.zIndex = '10000';
  
  document.body.appendChild(popupElement);
  
  // Position popup
  const range = getSelectionRange();
  if (range) {
    selectedTextRange = range;
    const position = getPopupPosition(range);
    popupElement.style.top = `${position.top}px`;
    popupElement.style.left = `${position.left}px`;
  } else {
    // Fallback positioning
    popupElement.style.top = '100px';
    popupElement.style.left = '100px';
  }

  // Close popup on outside click
  document.addEventListener('click', handleOutsideClick, true);
  
  // Close popup on escape key
  document.addEventListener('keydown', handleEscapeKey);
  
  console.log('Popup created at position:', popupElement.style.top, popupElement.style.left);
}

function handleOutsideClick(event) {
  if (popupElement && !popupElement.contains(event.target)) {
    closePopup();
  }
}

function handleEscapeKey(event) {
  if (event.key === 'Escape' && popupElement) {
    closePopup();
  }
}

function closePopup() {
  if (popupElement) {
    popupElement.remove();
    popupElement = null;
    selectedTextRange = null;
    document.removeEventListener('click', handleOutsideClick, true);
    document.removeEventListener('keydown', handleEscapeKey);
  }
}

function showLoadingPopup(selectedText) {
  createPopup();
  
  popupElement.innerHTML = `
    <div class="verifai-popup-header">
      <h3>VerifAI</h3>
      <button class="verifai-close-btn" onclick="document.getElementById('verifai-popup').remove()">×</button>
    </div>
    <div class="verifai-popup-body">
      <div class="verifai-loading">
        <div class="verifai-spinner"></div>
        <p>Аналізуємо текст...</p>
      </div>
    </div>
  `;
  
  // Attach close button handler
  const closeBtn = popupElement.querySelector('.verifai-close-btn');
  if (closeBtn) {
    closeBtn.addEventListener('click', closePopup);
  }
}

function showErrorPopup(error) {
  if (!popupElement) {
    createPopup();
  }
  
  popupElement.innerHTML = `
    <div class="verifai-popup-header">
      <h3>VerifAI</h3>
      <button class="verifai-close-btn">×</button>
    </div>
    <div class="verifai-popup-body">
      <div class="verifai-error">
        <p><strong>Помилка:</strong></p>
        <p>${escapeHtml(error)}</p>
        <p class="verifai-error-hint">Перевірте, чи запущений API сервер на http://localhost:8000</p>
      </div>
    </div>
  `;
  
  const closeBtn = popupElement.querySelector('.verifai-close-btn');
  if (closeBtn) {
    closeBtn.addEventListener('click', closePopup);
  }
}

function showResultsPopup(data, selectedText) {
  console.log('[Content] showResultsPopup called');
  console.log('[Content] Data received:', data);
  console.log('[Content] Selected text:', selectedText);
  console.log('[Content] Current popup element:', popupElement);
  
  // Always ensure popup exists
  if (!popupElement) {
    console.log('[Content] Creating new popup for results');
    createPopup();
  } else {
    console.log('[Content] Updating existing popup');
    console.log('[Content] Popup element HTML before update:', popupElement.innerHTML.substring(0, 100));
  }
  
  // Re-position popup if selection range is still available
  const range = getSelectionRange();
  if (range) {
    const position = getPopupPosition(range);
    popupElement.style.top = `${position.top}px`;
    popupElement.style.left = `${position.left}px`;
    console.log('[Content] Repositioned popup to:', position);
  }
  
  const manipulationStatus = data.manipulation;
  const statusClass = manipulationStatus ? 'verifai-manipulative' : 'verifai-not-manipulative';
  const statusText = manipulationStatus ? 'Маніпулятивний' : 'Не маніпулятивний';
  
  console.log('[Content] Manipulation status:', manipulationStatus);
  console.log('[Content] Status class:', statusClass);
  console.log('[Content] Techniques:', data.techniques);
  
  let techniquesHtml = '';
  if (data.techniques && data.techniques.length > 0) {
    techniquesHtml = `
      <div class="verifai-techniques">
        <h4>Техніки маніпуляції:</h4>
        ${data.techniques.map((tech, index) => {
          const name = getTechniqueName(tech);
          const description = getTechniqueFullDescription(tech);
          const techId = `technique-${index}`;
          return `
            <div class="verifai-technique-item">
              <div class="verifai-technique-name" data-technique-id="${techId}">
                <span>${escapeHtml(name)}</span>
                <span class="verifai-expand-icon">▼</span>
              </div>
              <div class="verifai-technique-description" id="${techId}-content">
                <p>${escapeHtml(description)}</p>
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
  } else {
    techniquesHtml = '<div class="verifai-techniques"><p>Техніки маніпуляції не виявлено</p></div>';
  }
  
  const explanationHtml = data.explanation ? `
    <div class="verifai-section">
      <div class="verifai-expandable-header" data-section="explanation">
        <h4>Пояснення</h4>
        <span class="verifai-expand-icon">▼</span>
      </div>
      <div class="verifai-expandable-content verifai-explanation-content" id="explanation-content">
        <div class="verifai-explanation-preview">
          <p>${escapeHtml(data.explanation.length > 150 ? data.explanation.substring(0, 150) + '...' : data.explanation)}</p>
        </div>
        <div class="verifai-explanation-full" style="display: none;">
          <p>${escapeHtml(data.explanation)}</p>
        </div>
      </div>
    </div>
  ` : '';
  
  const disinfoHtml = data.disinfo && data.disinfo.length > 0 ? `
    <div class="verifai-section">
      <div class="verifai-expandable-header" data-section="disinfo">
        <h4>Дебанк дезінформації (${data.disinfo.length})</h4>
        <span class="verifai-expand-icon">▼</span>
      </div>
      <div class="verifai-expandable-content" id="disinfo-content">
        <ol>
          ${data.disinfo.map(item => `<li>${escapeHtml(item)}</li>`).join('')}
        </ol>
      </div>
    </div>
  ` : '';
  
  console.log('[Content] Generated HTML lengths - techniques:', techniquesHtml.length, 'explanation:', explanationHtml.length, 'disinfo:', disinfoHtml.length);
  
  const newHtml = `
    <div class="verifai-popup-header">
      <h3>VerifAI</h3>
      <button class="verifai-close-btn">×</button>
    </div>
    <div class="verifai-popup-body">
      <div class="verifai-status ${statusClass}">
        ${statusText}
      </div>
      ${techniquesHtml}
      ${explanationHtml}
      ${disinfoHtml}
    </div>
  `;
  
  console.log('[Content] Setting innerHTML, length:', newHtml.length);
  popupElement.innerHTML = newHtml;
  console.log('[Content] innerHTML updated');
  
  // Attach event handlers
  const closeBtn = popupElement.querySelector('.verifai-close-btn');
  if (closeBtn) {
    closeBtn.addEventListener('click', closePopup);
    console.log('[Content] Close button attached');
  } else {
    console.warn('[Content] Close button not found!');
  }
  
  // Attach expandable section handlers
  const expandableHeaders = popupElement.querySelectorAll('.verifai-expandable-header');
  console.log('[Content] Found expandable headers:', expandableHeaders.length);
  expandableHeaders.forEach(header => {
    header.addEventListener('click', () => {
      const section = header.dataset.section;
      const content = popupElement.querySelector(`#${section}-content`);
      const icon = header.querySelector('.verifai-expand-icon');
      
      if (content) {
        const isExpanded = content.classList.contains('expanded');
        content.classList.toggle('expanded');
        icon.textContent = content.classList.contains('expanded') ? '▲' : '▼';
        
        // Handle explanation preview/full text toggle
        if (section === 'explanation') {
          const preview = content.querySelector('.verifai-explanation-preview');
          const full = content.querySelector('.verifai-explanation-full');
          if (preview && full) {
            if (!isExpanded) {
              preview.style.display = 'none';
              full.style.display = 'block';
            } else {
              preview.style.display = 'block';
              full.style.display = 'none';
            }
          }
        }
      }
    });
  });
  
  // Attach technique expand/collapse handlers
  const techniqueNames = popupElement.querySelectorAll('.verifai-technique-name');
  console.log('[Content] Found technique names:', techniqueNames.length);
  techniqueNames.forEach(nameElement => {
    nameElement.addEventListener('click', () => {
      const techniqueId = nameElement.dataset.techniqueId;
      const content = popupElement.querySelector(`#${techniqueId}-content`);
      const icon = nameElement.querySelector('.verifai-expand-icon');
      
      if (content) {
        content.classList.toggle('expanded');
        icon.textContent = content.classList.contains('expanded') ? '▲' : '▼';
      }
    });
  });
  
  // Force reflow and ensure visibility
  popupElement.style.display = 'block';
  popupElement.style.visibility = 'visible';
  popupElement.style.opacity = '1';
  // Force layout recalculation
  void popupElement.offsetHeight;
  
  // Small delay to ensure DOM update is processed
  setTimeout(() => {
    if (popupElement) {
      popupElement.style.display = 'block';
      popupElement.style.visibility = 'visible';
      console.log('[Content] Popup visibility enforced after timeout');
      console.log('[Content] Popup element in DOM:', document.body.contains(popupElement));
      console.log('[Content] Popup computed styles:', window.getComputedStyle(popupElement).display);
    } else {
      console.warn('[Content] Popup element disappeared!');
    }
  }, 10);
  
  console.log('[Content] showResultsPopup completed');
  console.log('[Content] Final popup HTML length:', popupElement.innerHTML.length);
}

function getTechniqueDescription(technique) {
  const descriptions = {
    "emotional_manipulation": "Емоційна маніпуляція - Використовує експресивну мову з сильним емоційним забарвленням або ейфорійний тон для підняття бойового духу та впливу на думку",
    "fear_appeals": "Апеляції до страху - Грає на страхах, стереотипах чи упередженнях. Включає тактики страху, невизначеності та сумнівів (FUD)",
    "bandwagon_effect": "Ефект натовпу - Використовує загальні позитивні концепції або заклики до мас ('всі так думають') для заохочення згоди",
    "selective_truth": "Селективна правда - Використовує логічні помилки, такі як вибірковий відбір фактів, whataboutism для відвернення критики, або створення опудальних аргументів",
    "cliche": "Думко-припиняючі кліше - Використовує формульні фрази, розроблені для припинення критичного мислення та завершення дискусії. Приклади: 'Все не так однозначно', 'Де ви були 8 років?'"
  };
  return descriptions[technique] || technique;
}

function getTechniqueName(technique) {
  const description = getTechniqueDescription(technique);
  const separatorIndex = description.indexOf(' - ');
  if (separatorIndex !== -1) {
    return description.substring(0, separatorIndex);
  }
  return description;
}

function getTechniqueFullDescription(technique) {
  const description = getTechniqueDescription(technique);
  const separatorIndex = description.indexOf(' - ');
  if (separatorIndex !== -1) {
    return description.substring(separatorIndex + 3);
  }
  return '';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

