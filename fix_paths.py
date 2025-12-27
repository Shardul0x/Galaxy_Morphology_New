# fix_all_paths.py
"""Fix all .pt to .pth and path separators"""

import os

files_to_fix = ['train.py', 'web_app.py', 'run_pipeline.py']

for filename in files_to_fix:
    if not os.path.exists(filename):
        print(f"⚠ {filename} not found, skipping")
        continue
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix file extensions
    content = content.replace("'best_vae.pt'", "'best_vae.pth'")
    content = content.replace("'best_pinn.pt'", "'best_pinn.pth'")
    content = content.replace('/best_vae.pt', '/best_vae.pth')
    content = content.replace('/best_pinn.pt', '/best_pinn.pth')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Fixed {filename}")

print("\n✅ All files updated!")
print("Now run: streamlit run web_app.py")
