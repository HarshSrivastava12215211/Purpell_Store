import os
import re

def remove_emojis(text):
    # Match emojis and other non-ASCII symbols, but keep regular punctuation, newlines, tabs, and common unicode like em-dash
    return re.sub(r'[^\x00-\x7F\u2014]', '', text)

files = []
for root, _, filenames in os.walk('.'):
    if 'node_modules' in root or '.venv' in root or '.git' in root:
        continue
    for f in filenames:
        if f.endswith('.html') or f.endswith('.py') or f.endswith('.md'):
            files.append(os.path.join(root, f))

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    new_content = remove_emojis(content)
    
    if new_content != content:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print(f"Removed emojis from {f}")
