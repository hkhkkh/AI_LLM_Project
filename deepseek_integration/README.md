# DeepSeek API Integration Project

This project integrates the DeepSeek API for testing and prompt engineering.

## Structure

- `api_client.py`: The main client class `DeepSeekClient` for interacting with the API. It handles authentication and session management.
- `config.py`: Configuration loader. Reads from `.env`.
- `promote_engineering.py`: A script to run test scenarios and refine prompts.
- `.env`: Stores your API Key. **Do not commit this file to version control.**

## Setup

1. Ensure you have `requests` installed:
   ```bash
   pip install requests
   ```
2. The `.env` file is already configured with your API key.

## Usage

Run the prompt engineering test script:
```bash
python promote_engineering.py
```

To use the client in your own scripts:
```python
from api_client import DeepSeekClient

client = DeepSeekClient()
response = client.simple_chat("Hello!")
print(response)
```
