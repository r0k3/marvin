with open("src/marvin/worker.py", "r") as f:
    content = f.read()

# We need to add reflective memory writing to the handle_sleep worker loop
old_write = """        # Write new Procedural rules
        for item in result.get("procedural", []):
            service.store_procedure(
                title=item.get("title", "New Rule"), steps=[item.get("rule", "")]
            )"""

new_write = """        # Write new Procedural rules
        for item in result.get("procedural", []):
            service.store_procedure(
                title=item.get("title", "New Rule"), steps=[item.get("rule", "")]
            )
            
        # Write new Reflective insights
        for item in result.get("reflective", []):
            service.reflect(
                title=item.get("title", "New Insight"), insight=item.get("insight", "")
            )"""

content = content.replace(old_write, new_write)
with open("src/marvin/worker.py", "w") as f:
    f.write(content)
