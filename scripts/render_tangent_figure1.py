"""Render Figure 1 as analytic vector geometry (no empirical data).

Run with system Python (ReportLab installed); pdftoppm supplies a 300-dpi PNG.
All three panels use the same orthographic camera and orthonormal block basis.
"""
from pathlib import Path
import math
import subprocess
from reportlab.graphics.shapes import Drawing, Line, Circle, Polygon, PolyLine, String
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/research/figures'
STEM = 'figure-1-factor-geometry'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
pdfmetrics.registerFont(TTFont('FigureSerif', FONT))
INK, GREY = '#222222', '#858585'
BLUE, TEAL, ORANGE = '#315a80', '#16776c', '#a65d12'
W, H, R = 540, 232, 48
D = Drawing(W, H)
ELEV, AZIM = math.radians(22), math.radians(-55)
VIEW = (math.cos(ELEV)*math.cos(AZIM), math.cos(ELEV)*math.sin(AZIM), math.sin(ELEV))
RIGHT = (-math.sin(AZIM), math.cos(AZIM), 0)
UP = (-math.sin(ELEV)*math.cos(AZIM), -math.sin(ELEV)*math.sin(AZIM), math.cos(ELEV))

def dot(a,b): return sum(x*y for x,y in zip(a,b))
def proj(p,cx): return cx + R*dot(p, RIGHT), 117 + R*dot(p, UP)
def text(x,y,s,size=8,color=INK,anchor='middle'):
    D.add(String(x,y,s,fontName='FigureSerif',fontSize=size,fillColor=HexColor(color),textAnchor=anchor))
def line(a,b,color=INK,width=.6,dash=None):
    D.add(Line(*a,*b,strokeColor=HexColor(color),strokeWidth=width,strokeDashArray=dash))
def arrow(a,b,color=INK,width=.8):
    line(a,b,color,width)
    dx,dy=b[0]-a[0],b[1]-a[1]; l=math.hypot(dx,dy); ux,uy=dx/l,dy/l
    D.add(Polygon([b[0],b[1],b[0]-4*ux+1.5*uy,b[1]-4*uy-1.5*ux,b[0]-4*ux-1.5*uy,b[1]-4*uy+1.5*ux],fillColor=HexColor(color),strokeColor=None))
def path(points,cx,color,width=.45,dash=None):
    D.add(PolyLine([v for p in points for v in proj(p,cx)],strokeColor=HexColor(color),strokeWidth=width,strokeDashArray=dash,fillColor=None))
def sphere(cx,ghost=False):
    D.add(Circle(cx,117,R,fillColor=HexColor('#fafbfc' if ghost else '#f1f7f5'),strokeColor=HexColor('#cbd5df' if ghost else TEAL),strokeWidth=.75))
    curves=[]
    for latitude in [-60,-30,0,30,60]:
        b=math.radians(latitude)
        curves.append([(math.cos(b)*math.cos(t),math.cos(b)*math.sin(t),math.sin(b)) for t in [2*math.pi*k/180 for k in range(181)]])
    for longitude in [0,45,90,135]:
        a=math.radians(longitude)
        curves.append([(math.cos(t)*math.cos(a),math.cos(t)*math.sin(a),math.sin(t)) for t in [2*math.pi*k/180 for k in range(181)]])
    for curve in curves:
        # Individually draw front/back segments to keep hidden geometry subdued.
        for a,b in zip(curve,curve[1:]):
            front=dot(a,VIEW)>0 and dot(b,VIEW)>0
            line(proj(a,cx),proj(b,cx), '#dce3e8' if ghost else (TEAL if front else '#d1e2de'), .32 if front else .22)

def axes(cx, color=GREY):
    for i,label in enumerate(['z₁','z₂','z₃']):
        p=[0,0,0]; p[i]=1.48
        a,b=proj((0,0,0),cx),proj(p,cx)
        arrow(a,b,color,.55)
        text(b[0]+(5 if i<2 else 0),b[1]+(-9 if i<2 else 5),label,8,color)

for cx,label,title in [(90,'(a)','Block span'),(270,'(b)','Factor manifold'),(450,'(c)','Local tangent')]:
    text(cx-73,211,label,10,anchor='start')
    text(cx+2,211,title,10)

# (a): finite window of the full three-dimensional span.
cx=90
sphere(cx,ghost=True)
verts=[(x,y,z) for x in [-1.05,1.05] for y in [-1.05,1.05] for z in [-1.05,1.05]]
for i,a in enumerate(verts):
    for b in verts[i+1:]:
        if sum(x!=y for x,y in zip(a,b))==1:
            line(proj(a,cx),proj(b,cx),'#9eafbf',.55)
axes(cx,BLUE)
text(cx,35,'V = span(d₁, d₂, d₃)',9)
text(cx,19,'dim V = 3',8.5)

# (b): samples occupy the sphere surface, not the enclosed ball.
cx=270
sphere(cx)
for k in range(55):
    z=1-2*(k+.5)/55; a=k*math.pi*(3-math.sqrt(5))
    p=(math.sqrt(1-z*z)*math.cos(a),math.sqrt(1-z*z)*math.sin(a),z)
    if dot(p,VIEW)>.25 and k%2==0:
        x,y=proj(p,cx); D.add(Circle(x,y,1.6,fillColor=HexColor(TEAL),strokeColor=None))
text(cx,35,'M = {z ∈ V : ‖z‖² = 1}',9)
text(cx,19,'dim M = 2',8.5)

# (c): affine tangent plane displayed at p=(0,0,1).
cx=450
sphere(cx)
p=(0,0,1)
polygon=[(-.92,-.92,1),(.92,-.92,1),(.92,.92,1),(-.92,.92,1)]
D.add(Polygon([v for q in polygon for v in proj(q,cx)],fillColor=Color(.98,.91,.79,alpha=.6),strokeColor=HexColor(ORANGE),strokeWidth=.7))
line(proj((0,0,0),cx),proj(p,cx),GREY,.6,[2,2])
x,y=proj(p,cx)
D.add(Circle(x,y,2.3,fillColor=HexColor(ORANGE),strokeColor=None))
text(x-5,y+7,'p',9,anchor='end')
for q,label in [((.85,0,1),'v₁'),((0,.85,1),'v₂')]:
    tip=proj(q,cx); arrow((x,y),tip,ORANGE,1)
    text(tip[0]+4,tip[1]-9,label,9,ORANGE)
text(cx+36,184,'p + TₚM',9,ORANGE)
text(cx,35,'TₚM = {v ∈ V : p · v = 0}',9)
text(cx,19,'dim TₚM = 2',8.5)

OUT.mkdir(parents=True,exist_ok=True)
renderPDF.drawToFile(D,str(OUT/f'{STEM}.pdf'))
renderSVG.drawToFile(D,str(OUT/f'{STEM}.svg'))
subprocess.run(['pdftoppm','-singlefile','-scale-to','2250','-png',str(OUT/f'{STEM}.pdf'),str(OUT/STEM)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
print(OUT/f'{STEM}.pdf')
print(OUT/f'{STEM}.svg')
print(OUT/f'{STEM}.png')
