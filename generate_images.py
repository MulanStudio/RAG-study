from PIL import Image, ImageDraw, ImageFont
import os

def create_text_image(filename, text, size=(800, 400), bg_color="white", text_color="black"):
    img = Image.new('RGB', size, color=bg_color)
    d = ImageDraw.Draw(img)
    
    # Try to load a default font, otherwise use default
    try:
        # For macOS
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Draw text
    y = 20
    for line in text.split('\n'):
        if line.startswith("#"):
            d.text((20, y), line.replace("#", ""), fill=text_color, font=title_font)
            y += 40
        else:
            d.text((20, y), line, fill=text_color, font=font)
            y += 30
            
    img.save(filename)
    print(f"Generated: {filename}")

# Image 1: Rig Specifications
text1 = """# Technical Specifications: Deepwater Titan Rig
Manufacturer: Transocean
Max Drilling Depth: 40,000 ft
Water Depth Rating: 12,000 ft
Hook Load: 3.0 million lbs
Blowout Preventer (BOP): 20,000 psi
Status: Operational in Gulf of Mexico
"""
create_text_image("downloads/images/rig_spec.png", text1)

# Image 2: Safety Warning
text2 = """# SAFETY ALERT: H2S Gas Detected
Location: Well Site B-14
Date: 2023-10-15
Hazard Level: Critical (Red)
Required PPE: SCBA (Self-Contained Breathing Apparatus)
Action: Evacuate upwind immediately.
"""
create_text_image("downloads/images/safety_alert.png", text2, bg_color="yellow", text_color="red")

# Image 3: Market Share
text3 = """# 2024 Market Share Projection (Offshore Drilling)
1. Valaris: 18%
2. Transocean: 16%
3. Seadrill: 12%
4. Noble Corp: 10%
5. Others: 44%
Note: Consolidation is expected to continue.
"""
create_text_image("downloads/images/market_share.png", text3)
