from api_client import DeepSeekClient
import time

def run_prompt_test(scenario_name, system_prompt, test_cases):
    """
    Run a set of test cases against a specific system prompt.
    """
    client = DeepSeekClient()
    print(f"\n{'='*20} Testing Scenario: {scenario_name} {'='*20}")
    print(f"System Prompt: {system_prompt}\n")
    
    results = []
    
    for i, user_input in enumerate(test_cases):
        print(f"--- Test Case {i+1} ---")
        print(f"Input: {user_input}")
        
        start_time = time.time()
        response = client.simple_chat(user_input, system_prompt=system_prompt)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"Output: {response}")
        print(f"Time taken: {duration:.2f}s\n")
        
        results.append({
            "input": user_input,
            "output": response,
            "duration": duration
        })
        
    return results

if __name__ == "__main__":
    # Example Scenario 1: Code Assistant
    scenario_1_prompt = "You are an expert Python programmer. Provide concise, efficient code solutions."
    scenario_1_tests = [
        "Write a function to calculate the Fibonacci sequence.",
        "How do I read a JSON file in Python?"
    ]
    
    # Example Scenario 2: Creative Writer
    scenario_2_prompt = "You are a creative writer specializing in science fiction. Write short, engaging stories."
    scenario_2_tests = [
        "Write a 50-word story about a robot discovering emotions.",
        "Describe a futuristic city on Mars."
    ]

    # Run tests
    run_prompt_test("Python Expert", scenario_1_prompt, scenario_1_tests)
    run_prompt_test("Sci-Fi Writer", scenario_2_prompt, scenario_2_tests)
