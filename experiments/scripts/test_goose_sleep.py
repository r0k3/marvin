import subprocess

# We will start goose and tell it to use the new sleep trigger we just added.
result = subprocess.run(
    ["goose", "run", "--text", "Use marvin_trigger_sleep to start consolidation."],
    capture_output=True,
    text=True
)

print(result.stdout)
