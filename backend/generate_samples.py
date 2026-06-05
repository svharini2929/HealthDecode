import os
from PIL import Image, ImageDraw, ImageFont

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_reports')
os.makedirs(SAMPLE_DIR, exist_ok=True)

def create_text_image(filename, text, bg_color=(255, 255, 255), fg_color=(0, 0, 0), size=(800, 1000)):
    """Creates a mock scanned text document image."""
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to draw some structured blocks representing medical reports
    # Since fonts vary by system, we'll draw rectangles/lines to make it look like a report sheet,
    # and write simple text labels
    draw.rectangle([20, 20, size[0]-20, size[1]-20], outline=fg_color, width=2)
    draw.line([20, 80, size[0]-20, 80], fill=fg_color, width=2)
    
    # Draw simple title text
    draw.text((40, 40), "CITY HEALTH LABS - MEDICAL REPORT", fill=fg_color)
    
    # Draw body lines
    y = 120
    for line in text.split('\n'):
        line = line.strip()
        if line:
            draw.text((40, y), line, fill=fg_color)
            y += 25
            
    filepath = os.path.join(SAMPLE_DIR, filename)
    img.save(filepath)
    print(f"Created sample text image: {filepath}")

def create_scan_image(filename, label, base_color=(50, 50, 50), size=(600, 600)):
    """Creates a mock radiological scan image (X-Ray/MRI/Ultrasound)."""
    img = Image.new('RGB', size, color=base_color)
    draw = ImageDraw.Draw(img)
    
    # Draw circular gradients/patterns to simulate a radiological scan
    center = (size[0] // 2, size[1] // 2)
    for r in range(size[0] // 3, 0, -15):
        shade = base_color[0] + (255 - base_color[0]) * (size[0] // 3 - r) // (size[0] // 3)
        draw.ellipse(
            [center[0] - r, center[1] - r, center[0] + r, center[1] + r],
            outline=(shade, shade, shade),
            width=2
        )
        
    # Draw indicators
    draw.text((20, 20), "PATIENT: John Doe", fill=(200, 200, 200))
    draw.text((20, 40), f"SCAN TYPE: {label}", fill=(200, 200, 200))
    draw.text((size[0] - 100, 20), "R", fill=(255, 100, 100)) # Right indicator
    
    filepath = os.path.join(SAMPLE_DIR, filename)
    img.save(filepath)
    print(f"Created sample scan image: {filepath}")

if __name__ == "__main__":
    # 1. Blood Report Image
    blood_text = """
    Patient Name: John Doe  Age: 45
    Test Name              Result     Reference
    Hemoglobin             13.2 *     13.8 - 17.2 g/dL
    Fasting Glucose        126 *      70 - 100 mg/dL
    LDL Cholesterol        172 *      < 100 mg/dL
    TSH                    5.2 *      0.4 - 4.5 uIU/mL
    """
    create_text_image("blood_report.png", blood_text)
    
    # 2. Chest X-Ray Image
    create_scan_image("chest_xray.png", "Chest X-Ray (Posterior-Anterior)", base_color=(40, 40, 42))
    
    # 3. Brain MRI Image
    create_scan_image("brain_mri.png", "Brain MRI (FLAIR)", base_color=(20, 20, 22))
    
    # 4. Ultrasound Image
    create_scan_image("abdomen_ultrasound.png", "Abdominal USG (Gallbladder)", base_color=(10, 15, 20))
