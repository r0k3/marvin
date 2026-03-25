with open("docs/guide/case-study.md", "r") as f:
    text = f.read()

# Replace the specific experiment intro
old_intro = """We simulated a complete conversational flow where an agent is instructed to read Act I of Shakespeare's *A Midsummer Night's Dream* and interact with a user's instructions.

During the session, the agent performed three distinct tasks:
1. **Answered a question** about the plot (extracting the harsh marriage laws and the suitors).
2. **Saved a user preference** (the user's favorite character is Bottom).
3. **Stored a literary analysis procedure** based on the user's instructions (always map power dynamics first)."""

new_intro = """We simulated an extreme conversational flow where an agent is instructed to read massive chunks of Shakespeare's *A Midsummer Night's Dream* and autonomously decide what to remember.

During the session, the agent:
1. **Processed** thousands of tokens of raw text.
2. **Autonomously logged** three distinct episodic summaries of the opening act.
3. **Triggered Computational Sleep** to distill the narrative into permanent facts and processing rules without any human guidance."""

text = text.replace(old_intro, new_intro)

with open("docs/guide/case-study.md", "w") as f:
    f.write(text)
