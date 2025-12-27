# fix_model_loading.py
"""Fix model loading to handle checkpoints with metadata"""

import re

with open('web_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to find model loading
old_pattern = r"vae\.load_state_dict\(torch\.load\(str\(vae_path\), map_location=device\)\)"

new_code = """checkpoint = torch.load(str(vae_path), map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    vae.load_state_dict(checkpoint['model_state_dict'])
                else:
                    vae.load_state_dict(checkpoint)"""

content = re.sub(old_pattern, new_code, content)

# Also fix the other location
old_pattern2 = r"vae\.load_state_dict\(torch\.load\(str\(model_path\), map_location=device\)\)"

new_code2 = """checkpoint = torch.load(str(model_path), map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['model_state_dict'])
        else:
            vae.load_state_dict(checkpoint)"""

content = re.sub(old_pattern2, new_code2, content)

with open('web_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed model loading in web_app.py")
print("Now run: streamlit run web_app.py")
