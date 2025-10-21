from google import genai
from google.genai.types import HttpOptions

# import os

# print(os.environ.get("GOOGLE_API_KEY"))

client = genai.Client()
# client = genai.Client(
#     http_options=HttpOptions(api_version="v1"),
# )
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="1.Is there an easy access to free dataset with time series of stock prices? The more the better but the more stocks the better",
)
print(response.text)
# Example response:
# Okay, let's break down how AI works. It's a broad field, so I'll focus on the ...
#
# Here's a simplified overview:
# ...
