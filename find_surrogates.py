with open('app.py', 'rb') as f:
    raw = f.read()

# Decode with surrogateescape to find bad bytes
content = raw.decode('utf-8', errors='surrogateescape')

# Find all surrogate characters
bads = [(i, hex(ord(c))) for i, c in enumerate(content) if 0xD800 <= ord(c) <= 0xDFFF]
print(f"Found {len(bads)} surrogate chars")
for pos, code in bads[:20]:
    ctx = content[max(0,pos-40):pos+40]
    print(f"  pos={pos} code={code} context={repr(ctx)}")
