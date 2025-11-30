import os

# Configuration
project_dir = "."  # Current directory
output_file = "full_codebase.txt"
extensions_to_include = [".py", ".md", ".yaml", ".sbatch", ".sh"] # Add others if needed (e.g., .json)
ignore_dirs = {".git", "__pycache__", "env", ".idea", ".vscode", "logs"}

def is_elastica_path(path):
    """Check if a path is within the elastica package in .venv."""
    normalized = os.path.normpath(path)
    # Check if path contains .venv/site-packages/elastica
    if ".venv" in normalized and "site-packages" in normalized and "elastica" in normalized:
        # Ensure the order is correct: .venv -> ... -> site-packages -> ... -> elastica
        parts = normalized.split(os.sep)
        try:
            venv_idx = parts.index(".venv")
            site_packages_idx = parts.index("site-packages")
            elastica_idx = parts.index("elastica")
            # site-packages must come after .venv, and elastica must come after site-packages
            if venv_idx < site_packages_idx < elastica_idx:
                return True
        except ValueError:
            pass
    return False

def is_on_elastica_path(root):
    """Check if we're on a path that leads to or is within the elastica package."""
    normalized = os.path.normpath(root)
    
    # If not in .venv, always allow
    if ".venv" not in normalized:
        return True
    
    # If we're already in the elastica path, allow
    if is_elastica_path(normalized):
        return True
    
    # The target path structure is: .venv/lib/pythonX.X/site-packages/elastica
    # Check if current path is a prefix of this path
    parts = normalized.split(os.sep)
    
    try:
        venv_idx = parts.index(".venv")
        # After .venv, we need: lib -> pythonX.X -> site-packages -> elastica
        if len(parts) <= venv_idx + 1:
            # We're at .venv level, allow (need to go into lib)
            return True
        
        if parts[venv_idx + 1] != "lib":
            # We're in .venv but not in lib, stop
            return False
        
        if len(parts) <= venv_idx + 2:
            # We're at .venv/lib level, allow (need to go into pythonX.X)
            return True
        
        if not parts[venv_idx + 2].startswith("python"):
            # We're in lib but not in pythonX.X, stop
            return False
        
        if len(parts) <= venv_idx + 3:
            # We're at .venv/lib/pythonX.X level, allow (need to go into site-packages)
            return True
        
        if parts[venv_idx + 3] != "site-packages":
            # We're in pythonX.X but not in site-packages, stop
            return False
        
        if len(parts) <= venv_idx + 4:
            # We're at .venv/lib/pythonX.X/site-packages level, allow (need to go into elastica)
            return True
        
        if parts[venv_idx + 4] != "elastica":
            # We're in site-packages but not in elastica, stop
            return False
        
        # We're in elastica or deeper, allow
        return True
    except (ValueError, IndexError):
        return False

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk(project_dir):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        # If we're in .venv, only continue if we're on the elastica path
        if ".venv" in root:
            if not is_on_elastica_path(root):
                # Skip this entire branch - we've diverged from the elastica path
                dirs[:] = []
                continue
            
            # If we're at site-packages level, only keep elastica directory
            normalized_root = os.path.normpath(root)
            if "site-packages" in normalized_root and "elastica" not in normalized_root:
                # We're in site-packages but not in elastica yet
                # Only keep the elastica directory
                if "elastica" in dirs:
                    dirs[:] = ["elastica"]
                else:
                    dirs[:] = []
                    continue
        
        for file in files:
            if any(file.endswith(ext) for ext in extensions_to_include):
                file_path = os.path.join(root, file)
                
                # Double-check: if file is in .venv, it must be in elastica package
                if ".venv" in file_path and not is_elastica_path(file_path):
                    continue
                
                # Write a clear header for each file
                outfile.write(f"\n\n{'='*20}\nFILE: {file_path}\n{'='*20}\n\n")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"Error reading file: {e}")

print(f"Project merged into {output_file}")