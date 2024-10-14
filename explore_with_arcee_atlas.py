import argparse
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description="Run optillm with specified approach and message.")
    parser.add_argument("--approach", required=True, help="The optillm approach to use.")
    parser.add_argument("--message",  required=True, help="The message to send to the model.")
    args = parser.parse_args()

    optillm_approach = args.approach
    message = args.message

    # This is a dummy key that is compatible with the openai encription. 
    OPENAI_KEY = "sk-no_key"
    OPENAI_BASE_URL = "http://localhost:8000/v1"
    client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)

    response = client.chat.completions.create(
        model=f"{optillm_approach}-/workspace/Arcee-Atlas",  # Assuming OptILM uses this naming convention
        messages=[
            {
              "role": "user",
              "content": message
            }
            ],
    )

    print(response)

if __name__ == "__main__":
    main()