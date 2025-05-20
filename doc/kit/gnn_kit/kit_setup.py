# Standard installation (all features, including the kit-mcp server)
# pip install cased-kit

from kit import Repository

# Load a local repository
repo = Repository("/home/trim/Documents/GitHub/GeneralizedNotationNotation")

# Load a remote public GitHub repo
# repo = Repository("https://github.com/owner/repo")

# Explore the repo
print(repo.get_file_tree())
# Output: [{"path": "src/main.py", "is_dir": False, ...}, ...]

print(repo.extract_symbols('src/main.py'))
# Output: [{"name": "main", "type": "function", "file": "src/main.py", ...}, ...]

repo.write_index("repo_index.json")