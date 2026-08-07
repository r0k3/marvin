---
name: recall
description: This skill should be used when the user runs /marvin-memory:recall or asks "what do you remember about X", "check memory for X", "do we have anything stored on X", "what did we decide about X". Searches the Marvin vault and answers from stored memory.
argument-hint: <query>
allowed-tools: Bash(marvin *)
---

# Recall from memory

Answer the user's question from the Marvin vault, not from guesswork.

Input: `$ARGUMENTS` as the search query.

1. Search across all four memory types:
   ```bash
   marvin search "$ARGUMENTS" --fields title,kind,path,excerpt
   ```
2. Read the most relevant hit in full — `marvin read <path>` — and any close runner-up when the excerpts disagree.
3. Answer the user's question grounded in what was read: state the fact/procedure/insight, its memory kind, and when it was last updated if visible. Quote values exactly (hostnames, versions, commands).
4. On zero hits, say memory has nothing stored for that query — do not invent. Offer to store the answer once established: `marvin remember ...`.

Deprecated facts are excluded from search automatically; if the user asks about history ("what did it used to be"), read the concept note — deprecated values remain in its frontmatter with reasons.
