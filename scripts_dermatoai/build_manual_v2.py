"""Regenera el manual PDF desde manual_content.json (no modifica el repositorio original)."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable, Preformatted, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from dermatoai.settings import ROOT
TEAL=colors.HexColor('#126c73'); INK=colors.HexColor('#18323c'); MUTED=colors.HexColor('#526572')
class Flow(Flowable):
    def __init__(self):
        super().__init__()
        self.width=490
        self.height=185
    def draw(self):
        c=self.canv
        labels=[('PDF / TXT',5,135),('Extraer y dividir',174,135),('SQLite + páginas',343,135),
                ('Pregunta + contexto',5,70),('Buscar fuentes',174,70),('LLM + fragmentos',343,70),
                ('Respuesta y citas',174,5)]
        for text,x,y in labels:
            c.setFillColor(colors.HexColor('#eaf5f4'));c.setStrokeColor(TEAL);c.roundRect(x,y,140,38,7,fill=1,stroke=1)
            c.setFont('Helvetica',9);c.setFillColor(INK);c.drawCentredString(x+70,y+15,text)
        def line(x1,y1,x2,y2):
            c.setStrokeColor(TEAL);c.line(x1,y1,x2,y2)
            import math
            a=math.atan2(y2-y1,x2-x1)
            for b in (-.45,.45):c.line(x2,y2,x2-6*math.cos(a+b),y2-6*math.sin(a+b))
        line(145,154,174,154);line(314,154,343,154)
        c.setStrokeColor(TEAL);c.line(413,135,413,122);c.line(413,122,244,122);line(244,122,244,108)
        line(145,89,174,89);line(314,89,343,89)
        line(413,70,413,24);line(413,24,314,24)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'docs_dermatoai/v2/Manual_DermatoAI_v2.pdf');args=ap.parse_args()
    font_dir=Path('/usr/share/fonts/truetype/dejavu')
    if (font_dir/'DejaVuSans.ttf').exists():
        pdfmetrics.registerFont(TTFont('Body',str(font_dir/'DejaVuSans.ttf')))
        pdfmetrics.registerFont(TTFont('BodyBold',str(font_dir/'DejaVuSans-Bold.ttf')))
    else:
        # En Windows usa fuentes PDF base; no requiere descargar tipografías.
        from reportlab.pdfbase.pdfmetrics import Font
        pdfmetrics.registerFont(Font('Body','Helvetica','WinAnsiEncoding'))
        pdfmetrics.registerFont(Font('BodyBold','Helvetica-Bold','WinAnsiEncoding'))
    styles={
      'title':ParagraphStyle('Title',fontName='BodyBold',fontSize=22,leading=27,textColor=INK,spaceAfter=9),
      'lead':ParagraphStyle('Lead',fontName='Body',fontSize=10,leading=14,textColor=MUTED,spaceAfter=18),
      'p':ParagraphStyle('Body',fontName='Body',fontSize=9.3,leading=13.6,textColor=INK,spaceAfter=9),
      'h':ParagraphStyle('Heading',fontName='BodyBold',fontSize=11,leading=15,textColor=TEAL,spaceBefore=10,spaceAfter=6),
      'cell':ParagraphStyle('Cell',fontName='Body',fontSize=8.4,leading=11.6,textColor=INK),
      'th':ParagraphStyle('TH',fontName='BodyBold',fontSize=8.6,leading=12,textColor=colors.white),
      'code':ParagraphStyle('Code',fontName='Courier',fontSize=7.1,leading=10,textColor=INK,backColor=colors.HexColor('#f1f4f6'),leftIndent=8,rightIndent=6,spaceBefore=4,spaceAfter=10)}
    pages=json.loads((ROOT/'docs_dermatoai/v2/manual_content.json').read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    story=[]
    def para(text,style):return Paragraph(escape(text).replace('\n','<br/>'),styles[style])
    import textwrap
    for i,page in enumerate(pages):
        if i:story.append(PageBreak())
        story += [para(f'{i+1:02d}  /  '+page['title'],'title'),para(page['lead'],'lead')]
        for b in page['blocks']:
            kind=b['type']
            if kind in ('p','h'):story.append(para(b['text'],kind))
            elif kind=='flow':story += [Flow(),Spacer(1,12)]
            elif kind=='code':
                wrapped='\n'.join('\n'.join(textwrap.wrap(line,103,replace_whitespace=False,drop_whitespace=False,break_long_words=True,break_on_hyphens=False)) if line else '' for line in b['text'].splitlines())
                story.append(Preformatted(wrapped,styles['code']))
            elif kind=='table':
                n=len(b['headers']);widths=b.get('widths') or ([142,348] if n==2 else [112,146,232])
                rows=[[para(x,'th') for x in b['headers']]]+[[para(x,'cell') for x in row] for row in b['rows']]
                t=Table(rows,colWidths=widths,hAlign='LEFT',repeatRows=1)
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),TEAL),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f6f6'),colors.white]),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.5,TEAL)]))
                story += [t,Spacer(1,9)]
    def footer(c,doc):
        c.setStrokeColor(TEAL);c.setLineWidth(1);c.line(52,795,543,795)
        c.setFont('Helvetica',8);c.setFillColor(MUTED);c.drawString(52,809,'DERMATOAI  /  MANUAL DE PREPARACION Y VALIDACION')
        c.drawString(52,29,'Version 2.1 · Uso academico · Revisar confirmaciones pendientes')
        c.drawRightString(543,29,str(doc.page))
    doc=SimpleDocTemplate(str(args.output),pagesize=(595.28,841.89),rightMargin=52,leftMargin=52,topMargin=62,bottomMargin=53,title='DermatoAI v2 - Preparacion, uso y validacion',author='Guia de la extension DermatoAI')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(args.output)
if __name__=='__main__':main()
