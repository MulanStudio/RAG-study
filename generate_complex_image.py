from PIL import Image, ImageDraw, ImageFont
import os

def create_complex_image(filename):
    # 创建一个白色背景图片
    img = Image.new('RGB', (1000, 800), color='white')
    d = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # 绘制标题
    d.text((50, 30), "Wellbore Schematic: ZT-09 (High Pressure High Temp)", fill='black', font=font_large)
    
    # 绘制井身结构示意图 (简单的矩形代表套管)
    # Conductor
    d.rectangle([300, 100, 700, 200], outline="black", width=3)
    d.text((720, 150), "Conductor: 30\" @ 300m", fill='blue', font=font_medium)
    
    # Surface Casing
    d.rectangle([350, 200, 650, 400], outline="black", width=3)
    d.text((670, 300), "Surface Casing: 20\" @ 1500m (K-55)", fill='blue', font=font_medium)
    
    # Intermediate Casing
    d.rectangle([400, 400, 600, 600], outline="black", width=3)
    d.text((620, 500), "Intermediate: 13-3/8\" @ 3500m (P-110)", fill='red', font=font_medium)
    
    # Production Liner
    d.rectangle([450, 600, 550, 750], outline="black", width=3)
    d.text((570, 680), "Prod Liner: 9-5/8\" @ 5200m (Q-125)", fill='red', font=font_medium)
    
    # 底部注释 (包含陷阱信息)
    note = """Warning: High H2S concentration detected at 4800m. 
    BOP Stack rated to 15,000 psi. 
    Mud Weight: 1.85 SG.
    Estimated Bottom Hole Temp: 185°C."""
    d.text((50, 700), note, fill='darkred', font=font_small)

    img.save(filename)
    print(f"Generated: {filename}")

if __name__ == "__main__":
    create_complex_image("downloads/challenge_set/Well_Schematic_ZT09.png")

