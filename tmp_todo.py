import re

with open("TO-DO.md", "r") as f:
    text = f.read()

# Replace trailing spaces in markdown table columns if present (from lints)
# but wait, the lints were for AGENTS.md and walkthrough.md, let's just 
# check what needs to be fixed.
