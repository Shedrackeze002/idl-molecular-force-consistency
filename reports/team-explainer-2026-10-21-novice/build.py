"""Render the source pages to a searchable A4 meeting brief."""
import argparse,json,re,html,math
from pathlib import Path
from reportlab.platypus import (BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,PageBreak,Flowable,KeepTogether)
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4

NAVY=colors.HexColor('#163447');TEAL=colors.HexColor('#176D73');INK=colors.HexColor('#203342');MUTED=colors.HexColor('#516774');PALE=colors.HexColor('#EEF5F6');RULE=colors.HexColor('#D5E0E5');GOLD=colors.HexColor('#BA6419')
FONTFILE=Path('/Library/Fonts/Arial Unicode.ttf')
if FONTFILE.exists():
 pdfmetrics.registerFont(TTFont('BodyFont',str(FONTFILE)));BODY='BodyFont'
else:BODY='Helvetica'
W,H=A4;M=43;CW=W-2*M
styles={
 'body':ParagraphStyle('body',fontName=BODY,fontSize=10.3,leading=15.0,textColor=INK,spaceAfter=7.7),
 'h':ParagraphStyle('h',fontName='Helvetica-Bold',fontSize=11.8,leading=15.4,textColor=NAVY,spaceBefore=6,spaceAfter=6),
 'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=22,leading=26,textColor=NAVY,spaceAfter=13),
 'kicker':ParagraphStyle('kicker',fontName='Helvetica-Bold',fontSize=8.5,leading=12,textColor=TEAL,spaceAfter=7),
 'source':ParagraphStyle('source',fontName=BODY,fontSize=8.1,leading=11,textColor=MUTED,spaceBefore=5,spaceAfter=0),
 'cell':ParagraphStyle('cell',fontName=BODY,fontSize=9.1,leading=12.7,textColor=INK),
 'th':ParagraphStyle('th',fontName='Helvetica-Bold',fontSize=8.9,leading=12.1,textColor=colors.white),
 'calltitle':ParagraphStyle('calltitle',fontName='Helvetica-Bold',fontSize=9,leading=12,textColor=TEAL,spaceAfter=5),
 'call':ParagraphStyle('call',fontName=BODY,fontSize=10.2,leading=15,textColor=NAVY),
 'bullet':ParagraphStyle('bullet',fontName=BODY,fontSize=10.1,leading=14.5,textColor=INK,leftIndent=12,firstLineIndent=-10,spaceAfter=6)
}
LINKS={1:'https://github.com/Shedrackeze002/idl-molecular-force-consistency/blob/3fa5969e5a0c2fdeaebc76a05bace20799715203/proposal.tex',15:'https://doi.org/10.1038/s41524-025-01878-x',16:'https://arxiv.org/abs/2401.07595',17:'https://openreview.net/forum?id=A8pqQipwkt',18:'https://proceedings.mlr.press/v267/bigi25a.html',19:'https://proceedings.mlr.press/v267/takamoto25a.html',20:'https://arxiv.org/abs/2007.09593',21:'https://github.com/janosh/matbench-discovery',24:'https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf'}
def markup(text):
 s=html.escape(str(text))
 return re.sub(r'\[(\d+)\]',lambda m:f'<link href="#ref{m[1]}" color="#176D73">[{m[1]}]</link>',s)
def par(text,style='body'):return Paragraph(markup(text),styles[style])
class Flow(Flowable):
 def __init__(self,nodes):Flowable.__init__(self);self.nodes=nodes;self.width=CW;self.height=76
 def draw(self):
  c=self.canv;n=len(self.nodes);gap=14;bw=(CW-gap*(n-1))/n
  for i,text in enumerate(self.nodes):
   x=i*(bw+gap);c.setFillColor(PALE);c.setStrokeColor(RULE);c.roundRect(x,16,bw,46,5,fill=1,stroke=1)
   c.setFont('Helvetica-Bold',9.1);c.setFillColor(NAVY);words=text.split();lines=[' '.join(words)]
   if c.stringWidth(lines[0],'Helvetica-Bold',9.1)>bw-10:lines=[words[0],' '.join(words[1:])]
   for k,t in enumerate(lines):c.drawCentredString(x+bw/2,42+(len(lines)-1)*5-k*12,t)
   if i<n-1:
    c.setStrokeColor(TEAL);c.line(x+bw+3,39,x+bw+gap-3,39);c.line(x+bw+gap-7,42,x+bw+gap-3,39);c.line(x+bw+gap-7,36,x+bw+gap-3,39)
class Chart(Flowable):
 def __init__(self,data):Flowable.__init__(self);self.data=data;self.width=CW;self.height=137
 def draw(self):
  c=self.canv;left=62;right=CW-12;scale=(right-left)/30;lo=25;ybase=29
  c.setFont('Helvetica',8.2);c.setFillColor(MUTED)
  for tick in range(25,56,5):
   x=left+(tick-lo)*scale;c.setStrokeColor(RULE);c.line(x,ybase,x,115);c.drawCentredString(x,17,str(tick))
  for i,(name,values) in enumerate(self.data.items()):
   y=102-i*29;c.setFillColor(INK);c.setFont('Helvetica-Bold',9.3);c.drawString(0,y-2,name)
   for j,vals in enumerate(values):
    yy=y+(4 if j==0 else -5);color=TEAL if j==0 else GOLD
    c.setStrokeColor(color);c.setFillColor(color);a,b=min(vals),max(vals);mean=sum(vals)/2
    x1=left+(a-lo)*scale;x2=left+(b-lo)*scale;xm=left+(mean-lo)*scale
    c.setLineWidth(1.4);c.line(x1,yy,x2,yy);c.line(x1,yy-3,x1,yy+3);c.line(x2,yy-3,x2,yy+3);c.circle(xm,yy,2.8,fill=1,stroke=0)
  c.setFillColor(MUTED);c.setFont('Helvetica',8.0);c.drawString(62,1,'Force MAE (meV/angstrom). Lines join the two run scores; they are not uncertainty limits.')
  c.setFillColor(TEAL);c.circle(CW-162,128,2.6,fill=1,stroke=0);c.setFont('Helvetica',8.4);c.drawString(CW-156,125,'BEST')
  c.setFillColor(GOLD);c.circle(CW-99,128,2.6,fill=1,stroke=0);c.drawString(CW-93,125,'Late20')
class Landscape(Flowable):
 def __init__(self):Flowable.__init__(self);self.width=CW;self.height=142
 def draw(self):
  c=self.canv;c.setStrokeColor(RULE);c.setLineWidth(.7);c.line(32,29,CW-12,29);c.line(32,29,32,133)
  c.setFillColor(MUTED);c.setFont('Helvetica',8.5);c.saveState();c.translate(17,82);c.rotate(90);c.drawCentredString(0,0,'Energy');c.restoreState()
  path=c.beginPath();path.moveTo(43,105);path.curveTo(74,45,109,36,149,69);path.curveTo(188,101,219,115,255,115);path.curveTo(291,115,322,75,353,49);path.curveTo(385,25,429,48,473,103)
  c.setStrokeColor(TEAL);c.setLineWidth(2.2);c.drawPath(path)
  c.setFillColor(NAVY);c.setFont('Helvetica-Bold',9);c.drawCentredString(255,129,'Unstable pass');c.drawCentredString(105,37,'Minimum A');c.drawCentredString(388,35,'Minimum B')
  for x1,y1,x2,y2 in [(225,113,194,97),(286,111,314,87)]:
   c.setStrokeColor(GOLD);c.setLineWidth(1.4);c.line(x1,y1,x2,y2)
   angle=math.atan2(y2-y1,x2-x1)
   for side in (-.55,.55):c.line(x2,y2,x2-6*math.cos(angle+side),y2-6*math.sin(angle+side))
  c.setFont('Helvetica',8.0);c.setFillColor(MUTED);c.drawCentredString(CW/2,15,'Different atomic arrangements along a schematic path')
  c.setFont('Helvetica',7.8);c.drawString(32,0,'Illustration only. Vertical height means energy, not physical height. No measured values are shown.')

class Doc(BaseDocTemplate):
 def afterFlowable(self,f):
  if getattr(f,'is_page_title',False):self.canv.bookmarkPage('page'+str(self.page));self.canv.addOutlineEntry(f.getPlainText(),'page'+str(self.page),level=0,closed=False)
def build(source,output):
 obj=json.loads(Path(source).read_text());total=len(obj['pages']);Path(output).parent.mkdir(parents=True,exist_ok=True)
 doc=Doc(str(output),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=64,bottomMargin=49,title=obj['title']+' | Project Explainer',author='Shedrack Eze | Group 9',subject='Landscape17 research evidence and team meeting preparation',pageCompression=1)
 frame=Frame(M,47,CW,H-108,id='main',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
 def decoration(c,d):
  c.saveState();c.setFillColor(NAVY);c.rect(0,H-9,W,9,fill=1,stroke=0);c.setFont('Helvetica-Bold',8);c.drawString(M,H-31,'LANDSCAPE17  |  GROUP 9');c.setFont('Helvetica',8);c.drawRightString(W-M,H-31,'MEETING BRIEF  |  21 OCTOBER 2026');c.setStrokeColor(RULE);c.line(M,35,W-M,35);c.setFont('Helvetica',7.7);c.setFillColor(MUTED);c.drawString(M,22,'ASD-STE100 Issue 9 writing rules | No-background edition');c.drawRightString(W-M,22,f'{d.page:02d} / {total}');c.restoreState()
 doc.addPageTemplates(PageTemplate(id='normal',frames=frame,onPage=decoration));story=[]
 for no,page in enumerate(obj['pages'],1):
  if no>1:story.append(PageBreak())
  story.append(par(page['section'],'kicker'));title=par(page['title'],'title');title.is_page_title=True;story.append(title)
  for b in page['blocks']:
   t=b['type']
   if t in ('p','h','source'):story.append(par(b['text'],'body' if t=='p' else t))
   elif t=='bullets':
    for item in b['items']:story.append(par('• '+item,'bullet'))
   elif t=='callout':
    inner=[par(b['title'],'calltitle'),par(b['text'],'call')];tab=Table([[inner]],colWidths=[CW]);tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('BOX',(0,0),(-1,-1),.6,RULE),('LEFTPADDING',(0,0),(-1,-1),11),('RIGHTPADDING',(0,0),(-1,-1),11),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),10)]));story.extend([tab,Spacer(1,9)])
   elif t=='table':
    weights=b.get('widths') or [1]*len(b['headers']);ww=[CW*x/sum(weights) for x in weights];rows=[[par(x,'th') for x in b['headers']]]
    for row in b['rows']:
     cells=[]
     for col,v in enumerate(row):
      if page.get('kind')=='sources' and col==0 and re.search(r'\[\d+\]',str(v)):
       ids=[]
       for match in re.finditer(r'\[(\d+)\](?:-\[(\d+)\])?',str(v)):
        first=int(match[1]);last=int(match[2]) if match[2] else first;ids.extend(range(first,last+1))
       anchors=''.join(f'<a name="ref{n}"/>' for n in ids);label=html.escape(str(v));url=LINKS.get(ids[0]) if len(ids)==1 else None
       mark=anchors+(f'<link href="{html.escape(url)}" color="#176D73">{label}</link>' if url else label);cells.append(Paragraph(mark,styles['cell']))
      else:cells.append(par(v,'cell'))
     rows.append(cells)
    tab=Table(rows,colWidths=ww,repeatRows=1,hAlign='LEFT');tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F6F8')]),('LINEBELOW',(0,0),(-1,0),.5,NAVY),('LINEBELOW',(0,1),(-1,-1),.3,RULE),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));
    if page.get('compact'):tab.setStyle(TableStyle([('TOPPADDING',(0,0),(-1,-1),3.5),('BOTTOMPADDING',(0,0),(-1,-1),3.5)]))
    story.extend([tab,Spacer(1,9)])
   elif t=='landscape':story.extend([Landscape(),Spacer(1,6)])
   elif t=='flow':story.append(Flow(b['nodes']))
   elif t=='chart':story.extend([Chart(b['data']),Spacer(1,8)])
   else:raise ValueError(t)
 doc.build(story)
 return obj

def markdown(obj):
 lines=['# '+obj['title'],'',obj['subtitle'],'','Snapshot: '+obj['snapshot_kigali'],'','Language: '+obj['language_policy'],'']
 for i,p in enumerate(obj['pages'],1):
  lines.extend(['## Page '+str(i)+': '+p['title'],''])
  for b in p['blocks']:
   t=b['type']
   if t in ('p','source'):lines.extend([b['text'],''])
   elif t=='h':lines.extend(['### '+b['text'],''])
   elif t=='bullets':lines.extend(['- '+x for x in b['items']]+[''])
   elif t=='callout':lines.extend(['**'+b['title']+'**', '',b['text'],''])
   elif t=='table':
    lines.extend(['| '+' | '.join(b['headers'])+' |','| '+' | '.join('---' for _ in b['headers'])+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in b['rows']]+[''])
   elif t=='landscape':lines.extend(['Illustration: an energy landscape with two valleys and a pass. This is not measured aspirin data.',''])
   elif t=='flow':lines.extend([' → '.join(b['nodes']),''])
   elif t=='chart':lines.extend(['Chart: BEST and Late20 means, with the two-seed ranges. Numeric values are in the table above.',''])
 return '\n'.join(lines)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(__file__).with_name('brief.json'));ap.add_argument('--output',type=Path,default=Path(__file__).with_name('meeting-brief.pdf'));a=ap.parse_args();o=build(a.source,a.output);a.source.with_name('brief.md').write_text(markdown(o).rstrip()+'\n');print(a.output)
