import requests
import json
import time
from config import API_KEY, API_URL

class DeepSeekClient:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or API_KEY
        self.base_url = base_url or API_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def chat_completion(self, messages, model="deepseek-chat", temperature=0.7, max_tokens=2048, stream=False):
        """
        Send a chat completion request to the DeepSeek API.
        
        Args:
            messages (list): List of message dictionaries (role, content).
            model (str): Model name (default: deepseek-chat).
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens to generate.
            stream (bool): Whether to stream the response.
            
        Returns:
            dict: The API response.
        """
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            
            if stream:
                return response # Return the response object for iterating over lines
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
            print(f"API Request Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Error Details: {e.response.text}")
            return None

    def simple_chat(self, user_input, system_prompt="You are a helpful assistant."):
        """
        A simplified method for a single turn chat.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        response = self.chat_completion(messages)
        if response and 'choices' in response:
            return response['choices'][0]['message']['content']
        return None

if __name__ == "__main__":
    # Simple test
    client = DeepSeekClient()
    print("Testing DeepSeek API connection...")
    result = client.simple_chat("Hello, are you ready?")
    print(f"Response: {result}")
