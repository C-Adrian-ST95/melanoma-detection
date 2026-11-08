"""Preparación documental auditable; las reglas de calidad son heurísticas, no validación médica."""
from collections import Counter
import io
import math
import re
import unicodedata
from pathlib import Path
from .ingestion import clean

PIPELINE_VERSION = '2.1-structured-1'
CLASSES = ('ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK')
PATTERNS = {
 'ACK':r'\bACK\b|queratosis act[ií]nic|actinic keratos',
 'BCC':r'\bBCC\b|\bCBC\b|basocelular|c[eé]lulas basales|basal cell',
 'MEL':r'\bMEL\b|melanoma',
 'NEV':r'\bNEV\b|\bnev[ou]s?\b|\bnevos\b|melanocytic nev|\blunares?\b',
 'SCC':r'\bSCC\b|\bCEC\b|espinocelular|epidermoide|escamocelular|c[eé]lulas escamosas|squamous cell',
 'SEK':r'\bSEK\b|queratosis seborreic|seborrheic keratos'}


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKD',str(text).lower()) if not unicodedata.combining(c))


def categories_for(text):
    return [key for key,pattern in PATTERNS.items() if re.search(pattern,str(text),re.I)]


def language_for(text):
    words=re.findall(r'[a-z]+',normalize(text))
    es=sum(w in {'el','la','las','los','del','una','para','por','que','con','piel','lesion','lesiones','es','y'} for w in words)
    en=sum(w in {'the','and','with','of','skin','lesion','lesions','is','this','for','from','are','was'} for w in words)
    return 'es' if es>=2 and es>en else ('en' if en>=2 and en>es else 'und')


def _words_to_text(words):
    lines=[]
    for word in sorted(words,key=lambda w:(round(w['top']/3),w['x0'])):
        if not lines or abs(lines[-1][0]-word['top'])>3:
            lines.append([word['top'],[]])
        lines[-1][1].append(word)
    output=[]
    previous=None
    for top,items in lines:
        if previous is not None and top-previous>19:
            output.append('')
        output.append(' '.join(w['text'] for w in sorted(items,key=lambda w:w['x0'])))
        previous=top
    return '\n'.join(output)


def _pdf_page_text(page):
    """Reconoce un canal central limpio; si no hay evidencia suficiente conserva el orden normal."""
    words=page.extract_words(x_tolerance=2,y_tolerance=3)
    if not words:
        return '',False
    body=[w for w in words if 45<w['top']<page.height-40]
    options=[]
    for mid in range(int(page.width*.42),int(page.width*.58)+1,2):
        left=[w for w in body if w['x1']<mid-5]
        right=[w for w in body if w['x0']>mid+5]
        cross=[w for w in body if w['x0']<=mid+5 and w['x1']>=mid-5]
        if len(left)>=40 and len(right)>=40:
            options.append((len(cross)/max(1,len(body)),abs(mid-page.width/2),left,right,cross))
    if not options:
        return page.extract_text(x_tolerance=2,y_tolerance=3) or '',False
    _,_,left,right,cross=min(options,key=lambda x:(x[0],x[1]))
    # Requiere suficientes palabras a ambos lados y un canal central vacío.
    two_columns=len(left)>=40 and len(right)>=40 and len(cross)/max(1,len(body))<.025
    if not two_columns:
        return page.extract_text(x_tolerance=2,y_tolerance=3) or '',False
    header=[w for w in words if w['top']<=45]
    footer=[w for w in words if w['top']>=page.height-40]
    return '\n\n'.join(filter(None,(_words_to_text(header),_words_to_text(left),_words_to_text(right),_words_to_text(cross),_words_to_text(footer)))),True


def read_pages(path):
    path=Path(path)
    body=path.read_bytes()
    if len(body)>25*1024*1024:
        raise ValueError('Máximo 25 MiB por documento.')
    warnings=[]
    if path.suffix.lower()=='.pdf':
        if not body.lstrip().startswith(b'%PDF-'):
            raise ValueError('Use un PDF válido.')
        # pdfplumber conserva posiciones para reconocer columnas antes de unir párrafos.
        import pdfplumber
        try:
            with pdfplumber.open(io.BytesIO(body),password='') as pdf:
                if len(pdf.pages)>300:
                    raise ValueError('Máximo 300 páginas por PDF.')
                pages=[]
                for i,page in enumerate(pdf.pages,1):
                    text,columns=_pdf_page_text(page)
                    pages.append((i,text))
                    if columns:warnings.append(f'Página {i}: orden de dos columnas estimado; revisar visualmente.')
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError('No se pudo leer el PDF; compruebe integridad, contraseña y texto extraíble.') from exc
        if sum(len(clean(t)) for _,t in pages)<30:
            raise ValueError('No hay texto extraíble. Realice OCR y revise el resultado antes de indexar.')
        if any(len(clean(t))<30 for _,t in pages):
            warnings.append('Hay páginas sin texto suficiente; pueden requerir OCR. Se conservan sus números.')
    elif path.suffix.lower() in {'.txt','.md'}:
        try:text=body.decode('utf-8-sig')
        except UnicodeDecodeError as exc:raise ValueError('Guarde TXT/MD con codificación UTF-8.') from exc
        pages=[(None,text)]
        if len(clean(text))<30:raise ValueError('El texto debe tener al menos 30 caracteres.')
    else:
        raise ValueError('Use PDF, TXT o Markdown.')
    if sum(len(text) for _,text in pages)>2_000_000:
        raise ValueError('Máximo 2.000.000 de caracteres extraídos.')
    return pages,warnings


HEADINGS=r'(resumen|abstract|introducci[oó]n|definici[oó]n|generalidades|caracter[ií]sticas(?: cl[ií]nicas)?|cl[ií]nica|diagn[oó]stico(?: diferencial)?|dermatoscop[ií]a|discusi[oó]n|conclusiones?|limitaciones?|resultados|m[eé]todos?|metodolog[ií]a|tratamiento|referencias(?: bibliogr[aá]ficas)?|bibliograf[ií]a|references|conclusions?|introduction|discussion|results|clinical features|diagnosis)'


def heading_for(line):
    value=line.strip()
    if value.startswith('#'):
        return value.lstrip('# ').strip() if len(value)<160 else None
    value=re.sub(r'^\d+(?:\.\d+)*[.)]?\s+','',value).strip(': .')
    if len(value)<=100 and re.fullmatch(HEADINGS,value,re.I):return value
    if len(value)<=100 and len(value.split())>=2 and line.strip().isupper() and any(c.isalpha() for c in value):return value
    return None


def clean_lines(pages):
    edges=Counter()
    for _,text in pages:
        lines=[l.strip() for l in text.splitlines() if l.strip()]
        for line in set(lines[:2]+lines[-2:]):
            key=re.sub(r'\d+','#',normalize(line))
            if len(key)<=120:edges[key]+=1
    repeated={key for key,count in edges.items() if len(pages)>1 and count>=max(2,math.ceil(.6*len(pages)))}
    result=[]
    for page,text in pages:
        lines=[]
        original=text.splitlines()
        nonempty=[i for i,l in enumerate(original) if l.strip()]
        edge_indices=set(nonempty[:2]+nonempty[-2:])
        for i,line in enumerate(original):
            key=re.sub(r'\d+','#',normalize(line.strip()))
            if i in edge_indices and key in repeated and not re.fullmatch(HEADINGS,line.strip(),re.I):
                continue
            if re.fullmatch(r'\s*(?:p[aá]gina\s*)?\d+\s*',line,re.I):continue
            line=unicodedata.normalize('NFKC',line).replace('\u00ad','')
            line=''.join(c for c in line if c.isprintable() or c.isspace()).strip()
            lines.append(line)
        text='\n'.join(lines)
        text=re.sub(r'([A-Za-zÁÉÍÓÚÜÑáéíóúüñ])-\n([a-záéíóúüñ])',r'\1\2',text)
        result.append((page,text))
    return result


def sentence_chunks(text,size=1500,overlap=180):
    """Solapamiento de frases completas; una frase excepcionalmente larga se corta por palabras."""
    sentences=re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÜÑ0-9¿¡])',clean(text))
    pieces=[]
    for sentence in sentences:
        while len(sentence)>size:
            end=sentence.rfind(' ',0,size)
            if end<1:end=size
            pieces.append(sentence[:end]);sentence=sentence[end:].strip()
        if sentence:pieces.append(sentence)
    current=[]
    for piece in pieces:
        if current and len(' '.join(current+[piece]))>size:
            yield ' '.join(current)
            tail=[]
            for sentence in reversed(current):
                if len(' '.join([sentence]+tail))>overlap:break
                tail.insert(0,sentence)
            current=tail
            if len(' '.join(current+[piece]))>size:current=[]
        current.append(piece)
    if current:yield ' '.join(current)


def quality_for(text,section):
    reasons=[]
    if re.search(r'^(referencias|bibliografia|references)\b',normalize(section)):
        reasons.append('referencias_bibliograficas')
    chars=[c for c in text if not c.isspace()]
    digits=sum(c.isdigit() for c in chars)/max(1,len(chars))
    letters=sum(c.isalpha() for c in chars)/max(1,len(chars))
    if digits>.25 or (digits>.12 and letters<.65):
        reasons.append('tabla_o_estadisticas_densas')
    if '\ufffd' in text or letters<.4 or any(len(w)>60 for w in text.split()):
        reasons.append('extraccion_poco_legible')
    return ('review' if reasons else 'ready'),reasons


def prepare(path,meta):
    pages,warnings=read_pages(path)
    cleaned=clean_lines(pages)
    section='Sin sección identificada'
    blocks=[]
    for page,text in cleaned:
        buffer=[]
        def flush():
            if buffer:
                blocks.append((page,section,clean(' '.join(buffer))))
                buffer.clear()
        for line in text.splitlines():
            heading=heading_for(line)
            if heading:
                flush();section=heading
            elif not line.strip():flush()
            else:buffer.append(line)
        flush()
    parts=[]
    for page,section,text in blocks:
        for part in sentence_chunks(text):
            if len(part)<20:continue
            quality,reasons=quality_for(part,section)
            detected=categories_for(section+' '+part)
            categories=sorted(set(detected+meta.get('categories',[])))
            guessed=language_for(part)
            language=guessed if guessed!='und' else (meta.get('language','auto') if meta.get('language','auto')!='auto' else 'und')
            if meta.get('language') not in (None,'auto') and guessed not in ('und',meta['language']):
                warnings.append(f'Página {page or "TXT"}: posible mezcla de idiomas; revise el texto.')
            parts.append(dict(page=page,section=section,text=part,categories=categories,
                              language=language,quality=quality,reasons=reasons))
    if not parts or len(parts)>6000:raise ValueError('Documento vacío o más de 6.000 fragmentos.')
    report={'pipeline':PIPELINE_VERSION,'pages':len(pages),'ready':sum(p['quality']=='ready' for p in parts),
            'review':sum(p['quality']=='review' for p in parts),'warnings':list(dict.fromkeys(warnings))[:100],
            'category_note':'Categorías sugeridas por palabras y metadatos; requieren comprobación humana.'}
    return pages,parts,report
