import os

# Configuration
project_dir = "."  # Current directory
output_file = "full_codebase.txt"
extensions_to_include = [".py", ".md", ".yaml"] # Add others if needed (e.g., .json)
ignore_dirs = {".git", "__pycache__", "venv", "env", ".idea", ".vscode", "logs"}

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk(project_dir):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            if any(file.endswith(ext) for ext in extensions_to_include):
                file_path = os.path.join(root, file)
                # Write a clear header for each file
                outfile.write(f"\n\n{'='*20}\nFILE: {file_path}\n{'='*20}\n\n")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"Error reading file: {e}")

print(f"Project merged into {output_file}")