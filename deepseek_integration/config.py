import os

def load_env_file(env_path):
    """Simple .env file loader to avoid external dependencies"""
    if not os.path.exists(env_path):
        return
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Load environment variables from .env file in the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(current_dir, '.env')
load_env_file(env_file)

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com")

if not API_KEY:
    print("Warning: DEEPSEEK_API_KEY not found in environment variables or .env file.")
