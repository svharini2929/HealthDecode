import os
import uuid
import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from database import init_db, save_report, get_all_reports, get_report, delete_report
from ocr_service import extract_text_from_pdf, extract_text_from_image
from ai_service import analyze_report_with_ai

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for React frontend

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize SQLite database on startup
init_db()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.datetime.now().isoformat()})

@app.route('/api/analyze', methods=['POST'])
def analyze_report():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
        
    try:
        # Save uploaded file
        report_id = str(uuid.uuid4())
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()
        saved_filename = f"{report_id}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        # Determine file type
        file_type = "Image"
        if file_ext == ".pdf":
            file_type = "PDF"
            
        # Extract text based on file type
        print(f"Extracting text from {file_type}: {filename}...")
        if file_type == "PDF":
            extracted_text = extract_text_from_pdf(filepath)
            # If digital PDF extraction returned nothing, it might be a scanned PDF. Try OCR.
            if not extracted_text.strip():
                print("Digital PDF text extraction empty, attempting scanned PDF/OCR fallback...")
                extracted_text = extract_text_from_image(filepath, hint_filename=filename)
        else:
            extracted_text = extract_text_from_image(filepath, hint_filename=filename)
            
        if not extracted_text.strip():
            extracted_text = f"Scanned Medical Image/Report File: {filename}"
            
        # Run AI analysis
        print("Analyzing extracted medical text with AI...")
        analysis = analyze_report_with_ai(extracted_text)
        
        # Save to SQLite
        save_report(
            report_id=report_id,
            filename=filename,
            file_type=file_type,
            original_text=extracted_text,
            summary=analysis['summary'],
            observations=analysis['observations'],
            abnormal_values=analysis['abnormal_values'],
            terminology=analysis['terminology'],
            doctor_questions=analysis['doctor_questions']
        )
        
        # Return complete analysis response
        result = get_report(report_id)
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error analyzing report: {e}")
        return jsonify({'error': f"An error occurred during analysis: {str(e)}"}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        reports = get_all_reports()
        return jsonify(reports), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<report_id>', methods=['GET'])
def get_report_details(report_id):
    try:
        report = get_report(report_id)
        if not report:
            return jsonify({'error': 'Report not found'}), 404
        return jsonify(report), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<report_id>', methods=['DELETE'])
def delete_report_record(report_id):
    try:
        report = get_report(report_id)
        if not report:
            return jsonify({'error': 'Report not found'}), 404
            
        delete_report(report_id)
        return jsonify({'message': 'Report successfully deleted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/pdf/<report_id>', methods=['GET'])
def export_pdf(report_id):
    try:
        report = get_report(report_id)
        if not report:
            return jsonify({'error': 'Report not found'}), 404
            
        # Create PDF path
        pdf_filename = f"healthdecode_summary_{report_id}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        
        generate_pdf_report(report, pdf_path)
        
        return send_file(pdf_path, as_attachment=True, download_name=f"HealthDecode_{report['filename']}.pdf")
    except Exception as e:
        print(f"Error exporting PDF: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/sheet/<report_id>', methods=['GET'])
def export_doctor_sheet(report_id):
    try:
        report = get_report(report_id)
        if not report:
            return jsonify({'error': 'Report not found'}), 404
            
        # Create PDF path for Doctor Discussion Sheet
        sheet_filename = f"doctor_discussion_{report_id}.pdf"
        sheet_path = os.path.join(app.config['UPLOAD_FOLDER'], sheet_filename)
        
        generate_doctor_sheet(report, sheet_path)
        
        return send_file(sheet_path, as_attachment=True, download_name=f"Doctor_Prep_{report['filename']}.pdf")
    except Exception as e:
        print(f"Error exporting Doctor Sheet: {e}")
        return jsonify({'error': str(e)}), 500


# PDF Generation helpers using ReportLab
def generate_pdf_report(report, output_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    story = []
    
    # Custom Styles
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor("#0f766e")   # Deep Teal
    secondary_color = colors.HexColor("#0284c7") # Healthcare Blue
    text_color = colors.HexColor("#334155")      # Charcoal
    accent_color = colors.HexColor("#b91c1c")    # Crimson Red for alerts
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=primary_color,
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=primary_color,
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,
        leading=14,
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,
        leading=14,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=6
    )
    
    disclaimer_style = ParagraphStyle(
        'DocDisclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-BoldOblique',
        fontSize=9,
        textColor=colors.HexColor("#475569"),
        leading=13,
        spaceAfter=15
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.white,
        leading=11
    )
    
    table_body_style = ParagraphStyle(
        'TableBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=text_color,
        leading=12
    )

    # 1. Header & Title
    story.append(Paragraph("HealthDecode", title_style))
    story.append(Paragraph("Medical Second-Opinion Explanation & Educational Guide", ParagraphStyle('SubTitle', fontName='Helvetica-Bold', fontSize=12, textColor=secondary_color, spaceAfter=10)))
    story.append(Spacer(1, 10))
    
    # 2. Safety Disclaimer Banner (Crucial Risk Layer)
    disclaimer_text = (
        "<b>IMPORTANT NOTICE:</b> This tool is for educational purposes only and is not a substitute for professional "
        "medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider "
        "with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay "
        "seeking it because of something you have read in this document."
    )
    
    disclaimer_table = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[500])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 15))
    
    # Metadata Table
    meta_data = [
        [Paragraph(f"<b>Uploaded Document:</b> {report['filename']}", body_style), 
         Paragraph(f"<b>Generated On:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[250, 250])
    meta_table.setStyle(TableStyle([
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))
    
    # 3. Patient-Friendly Summary
    story.append(Paragraph("Patient-Friendly Summary", h2_style))
    story.append(Paragraph(report['summary'], body_style))
    story.append(Spacer(1, 10))
    
    # 4. Key Observations
    story.append(Paragraph("Key Observations & Findings", h2_style))
    for obs in report['observations']:
        obs_text = f"<b>{obs['title']}:</b> {obs['description']}"
        story.append(Paragraph(f"&bull; {obs_text}", bullet_style))
    story.append(Spacer(1, 10))
    
    # 5. Abnormal Values (Highlighted Table)
    if report['abnormal_values']:
        story.append(Paragraph("Values Outside Reference Ranges", h2_style))
        table_data = [[
            Paragraph("Parameter", table_header_style), 
            Paragraph("Value", table_header_style), 
            Paragraph("Reference Range", table_header_style), 
            Paragraph("Explanation", table_header_style)
        ]]
        
        for val in report['abnormal_values']:
            table_data.append([
                Paragraph(f"<b>{val['parameter']}</b>", table_body_style),
                Paragraph(val['value'], table_body_style),
                Paragraph(val['reference_range'], table_body_style),
                Paragraph(val['interpretation'], table_body_style)
            ])
            
        val_table = Table(table_data, colWidths=[110, 80, 100, 210])
        val_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(val_table)
        story.append(Spacer(1, 10))
        
    doc.build(story)


def generate_doctor_sheet(report, output_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    story = []
    
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor("#0f766e")
    secondary_color = colors.HexColor("#0284c7")
    text_color = colors.HexColor("#334155")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=primary_color,
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,
        leading=14,
        spaceAfter=8
    )
    
    question_style = ParagraphStyle(
        'DocQuestion',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        textColor=secondary_color,
        leading=14,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,
        leading=14,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=6
    )

    disclaimer_style = ParagraphStyle(
        'DocDisclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        textColor=colors.HexColor("#64748b"),
        leading=12,
        spaceAfter=10
    )

    # Header
    story.append(Paragraph("Doctor Visit Discussion Guide", title_style))
    story.append(Paragraph("Prepared by HealthDecode - Patient Education Companion", ParagraphStyle('SubTitle', fontName='Helvetica-Bold', fontSize=10, textColor=secondary_color, spaceAfter=8)))
    story.append(Spacer(1, 10))
    
    # Meta
    meta_text = f"<b>Patient Guide For Report:</b> {report['filename']} | <b>Created:</b> {datetime.datetime.now().strftime('%Y-%m-%d')}"
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 5))
    
    disclaimer_text = (
        "This sheet is designed to help you organize your thoughts and prepare questions for your consulting doctor. "
        "It is for educational purposes and should not be used as a medical diagnosis or record."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    story.append(Spacer(1, 10))
    
    # Symptoms & Observations Checklist
    story.append(Paragraph("1. Key Findings to Bring Up", h2_style))
    story.append(Paragraph("Make sure to check if your physician has reviewed these specific points from your report:", body_style))
    for obs in report['observations']:
        story.append(Paragraph(f"[  ] <b>{obs['title']}</b> - {obs['description']}", bullet_style))
    
    if report['abnormal_values']:
        for val in report['abnormal_values']:
            story.append(Paragraph(f"[  ] <b>{val['parameter']}</b> was noted as {val['value']} (Normal range: {val['reference_range']})", bullet_style))
            
    story.append(Spacer(1, 10))
    
    # Prepared Questions Section
    story.append(Paragraph("2. Questions for Your Doctor", h2_style))
    story.append(Paragraph("Here are 4-5 structured questions customized to your report that you can ask your doctor:", body_style))
    
    for idx, question in enumerate(report['doctor_questions'], 1):
        story.append(Paragraph(f"Question {idx}:", question_style))
        story.append(Paragraph(question, body_style))
        
    story.append(Spacer(1, 10))
    
    # Notes Section for Patient (Printable lined spacer)
    story.append(Paragraph("3. Symptoms or Notes to Mention", h2_style))
    story.append(Paragraph("Use this section to write down how you've been feeling, any symptoms you have (e.g. fatigue, pain, cough), or any medications you are currently taking, so you don't forget to mention them:", body_style))
    story.append(Spacer(1, 15))
    
    # Lined spacing
    lines_table = Table([[""]]*4, colWidths=[500], rowHeights=[20]*4)
    lines_table.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(lines_table)
    
    doc.build(story)


if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
