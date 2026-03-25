import os
from litellm import completion

os.environ.setdefault("OPENAI_API_KEY", "your-api-key-here")

prompt = '''
You are an AI agent's memory consolidation worker (computational sleep).
Analyze the following raw episodic logs from a recent coding session.

Your goal is to extract:
1. "semantic": Permanent architectural facts, decisions, or user preferences.
2. "procedural": Reusable coding rules, steps, or conventions to prevent future errors.
3. "reflective": High-level insights, patterns, or principles realized from the logs.

Output valid JSON ONLY in this exact format:
{
  "semantic": [
    {"concept": "...", "fact": "..."}
  ],
  "procedural": [
    {"title": "...", "rule": "..."}
  ],
  "reflective": [
    {"title": "...", "insight": "..."}
  ]
}

Raw Episodes:
1. The user told me their favorite character is Bottom.
2. The user said: whenever we analyze a play, we must map out power dynamics before looking at character emotions.
3. I realized that the law of Athens acts as the primary antagonist driving the characters into the chaotic forest.
'''

response = completion(model='gpt-5.4', messages=[{'role': 'user', 'content': prompt}], response_format={'type': 'json_object'})
print(response.choices[0].message.content)
