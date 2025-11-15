#!/usr/bin/env python3
"""
Generate placeholder icons for VerifAI extension.
Requires PIL/Pillow: pip install pillow
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create a simple colored square icon"""
    # Create image with transparent background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw background circle
    margin = size // 8
    draw.ellipse([margin, margin, size - margin, size - margin], 
                 fill=(0, 123, 255, 255))  # Blue color
    
    # Try to add text "V" if font is available
    try:
        font_size = size // 2
        # Try to use a default font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Calculate text position (centered)
        text = "V"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - bbox[1]
        
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    except Exception as e:
        print(f"Warning: Could not add text to icon: {e}")
    
    img.save(output_path, 'PNG')
    print(f"Created {output_path} ({size}x{size})")

if __name__ == "__main__":
    icon_dir = os.path.dirname(os.path.abspath(__file__))
    
    sizes = [16, 48, 128]
    for size in sizes:
        output_path = os.path.join(icon_dir, f"icon{size}.png")
        create_icon(size, output_path)
    
    print("\nIcons created successfully!")
    print("If you have a custom icon design, replace these placeholder files.")

