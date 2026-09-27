import re

with open('app.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Fix 1: broken startSampleVideo line
broken = "await fetch('/api/start_sample_video', {        async function startIPCamera()"
fixed  = "await fetch('/api/start_sample_video', { method: 'POST' });\n        }\n\n        async function startIPCamera()"
content = content.replace(broken, fixed, 1)

# Fix 2: remove any leftover garbage lines like " false;\n            }\n        }"
content = re.sub(r'\n\s*false;\s*\n\s*\}\s*\n\s*\}\s*\n', '\n', content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done. Lines written:", content.count('\n'))
