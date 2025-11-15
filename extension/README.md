# VerifAI Browser Extension

Browser extension for verifying selected text using the VerifAI API.

## Features

- **Text Selection**: Select any text on a webpage
- **Context Menu**: Right-click and choose "Verify with VerifAI"
- **Analysis Results**: See manipulation status, techniques, explanations, and disinformation debunking
- **Expandable Sections**: Click to expand explanation and disinfo sections

## Installation

### Prerequisites

1. Make sure the VerifAI API server is running on `http://localhost:8000`
   - Start it with: `uv run scripts/start_api.py`

### Load Extension in Chrome

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `extension/` directory from this project
5. The extension should now be installed

## Usage

1. Select any text on a webpage
2. Right-click on the selected text
3. Choose "Verify with VerifAI" from the context menu
4. Wait for the analysis (a loading spinner will appear)
5. View the results in the popup:
   - **Status Tag**: Red (manipulative) or Green (not manipulative)
   - **Techniques**: List of detected manipulation techniques with Ukrainian descriptions
   - **Explanation**: Expandable section with detailed analysis
   - **Disinfo Debunk**: Expandable section with numbered counter-arguments

## Configuration

The API URL defaults to `http://localhost:8000/analyze`. To change it:

1. Open Chrome DevTools (F12)
2. Go to the Application/Storage tab
3. Navigate to Extensions → Storage → chrome-extension://[your-extension-id]
4. Add a key `apiUrl` with your desired URL value

Or modify the `DEFAULT_API_URL` constant in `background.js`.

## File Structure

```
extension/
├── manifest.json       # Extension configuration
├── background.js       # Service worker for API calls
├── content.js          # Content script for popup injection
├── popup.html          # Action popup HTML
├── popup.css           # Styles for injected popup
├── popup.js            # Action popup logic
├── icons/              # Extension icons
│   ├── icon16.png
│   ├── icon48.png
│   ├── icon128.png
│   └── generate_icons.py
└── README.md           # This file
```

## API Response Format

The extension expects the API to return:

```json
{
  "manipulation": true,
  "techniques": ["emotional_manipulation", "fear_appeals"],
  "explanation": "Detailed explanation text...",
  "disinfo": ["First debunk point", "Second debunk point"]
}
```

## Troubleshooting

- **Extension not working**: Make sure the API server is running on `http://localhost:8000`
- **CORS errors**: The API server should have CORS enabled (already configured in `api/main.py`)
- **Icons not showing**: Regenerate icons with `uv run python extension/icons/generate_icons.py`
- **Popup not appearing**: Check browser console for errors (F12 → Console tab)

## Development

To regenerate icons:
```bash
uv run python extension/icons/generate_icons.py
```

