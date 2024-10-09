import os
import sys
import argparse
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description="Run optillm with specified approach and message.")
    parser.add_argument("--approach", required=True, help="The optillm approach to use.")
    parser.add_argument("--message",  required=True, help="The message to send to the model.")
    args = parser.parse_args()

    optillm_approach = args.approach
    message = args.message

    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_BASE_URL = "http://47.47.180.6:11375/v1"
    client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)

    response = client.chat.completions.create(
      model="/workspace/Arcee-Atlas",
      messages=[
        {
          "role": "user",
          "content": f"<optillm_approach>{optillm_approach}</optillm_approach> {message}"
        }
      ],
      temperature=0.2,
    )

    print(response)

if __name__ == "__main__":
    main()