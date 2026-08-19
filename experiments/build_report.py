"""
Build the final report in three formats from one source.

    docs/report.html   self-contained page, figures embedded as data URIs
                       (this is what gets published as an artifact)
    docs/report.docx   Microsoft Word document, figures embedded
    docs/report.md     Markdown, figures referenced from docs/figures/

Source of truth is docs/report_template.html, which contains {{FIGn}}
placeholders. Run experiments/make_report_figures.py first to (re)generate
docs/figures/.

    python experiments/build_report.py
    python experiments/build_report.py --formats html docx

The .docx path uses python-docx if available. It is written by walking the
report's own HTML structure so the Word file carries the same headings, tables,
figures and captions rather than being a separate hand-maintained document.
"""
import argparse
import base64
import html.parser
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import repo_root

ROOT = repo_root()
DOCS = os.path.join(ROOT, 'docs')
FIGS = os.path.join(DOCS, 'figures')
TEMPLATE = os.path.join(DOCS, 'report_template.html')

# placeholder -> (filename in docs/figures, mime type)
FIGURES = {
    'FIG1': ('fig1_scene.jpg', 'image/jpeg'),
    'FIG2': ('fig2_timeseries.png', 'image/png'),
    'FIG3': ('fig3_metric_artifact.png', 'image/png'),
    'FIG4': ('fig4_polarity.png', 'image/png'),
    'FIG5': ('fig5_bands_budget.png', 'image/png'),
    'FIG6': ('fig6_wiener.png', 'image/png'),
    'FIG7': ('fig7_ablation.png', 'image/png'),
}


def read_template():
    if not os.path.exists(TEMPLATE):
        raise FileNotFoundError(f'missing {TEMPLATE}')
    with open(TEMPLATE, encoding='utf-8') as f:
        return f.read()


def check_figures():
    missing = [fn for fn, _ in FIGURES.values() if not os.path.exists(os.path.join(FIGS, fn))]
    if missing:
        raise FileNotFoundError(
            'missing figures: ' + ', '.join(missing) +
            '\n  run:  python experiments/make_report_figures.py')


# --------------------------------------------------------------------- HTML
def build_html():
    s = read_template()
    for key, (fn, mime) in FIGURES.items():
        with open(os.path.join(FIGS, fn), 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        s = s.replace('{{' + key + '}}', f'data:{mime};base64,{b64}')
    if '{{' in s:
        raise RuntimeError('unsubstituted placeholder remains in template')
    out = os.path.join(DOCS, 'report.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(s)
    return out


# ----------------------------------------------------------------- Markdown
def build_markdown():
    """Plain Markdown with figures referenced by relative path. Useful for
    GitHub rendering and as a fallback Word import route (Word opens .md via
    pandoc, or it can be pasted)."""
    s = read_template()
    for key, (fn, _) in FIGURES.items():
        s = s.replace('{{' + key + '}}', f'figures/{fn}')
    md = _HtmlToMarkdown()
    md.feed(s)
    out = os.path.join(DOCS, 'report.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md.result())
    return out


class _HtmlToMarkdown(html.parser.HTMLParser):
    """Minimal, purpose-built converter for this report's own markup only."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.skip = 0
        self.in_row = None
        self.in_cell = None
        self.header_row = False
        self.tbl_cols = 0
        self.list_stack = []
        self.pending = []
        self.fig_caption = False

    # -- helpers
    def _emit(self, text=''):
        self.out.append(text)

    def _flush(self, prefix='', suffix=''):
        txt = re.sub(r'\s+', ' ', ''.join(self.pending)).strip()
        self.pending = []
        if txt:
            self._emit(prefix + txt + suffix)
            self._emit('')
        return txt

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ('style', 'script'):
            self.skip += 1
        elif tag == 'title':
            self.skip += 1
        elif tag in ('h1', 'h2', 'h3', 'h4'):
            self._flush()
            self.pending = []
            self._level = {'h1': '# ', 'h2': '## ', 'h3': '### ', 'h4': '#### '}[tag]
        elif tag == 'img':
            src = a.get('src', '')
            alt = a.get('alt', 'figure')
            self._emit(f'![{alt}]({src})')
            self._emit('')
        elif tag == 'table':
            self._emit('')
            self.tbl_cols = 0
        elif tag == 'tr':
            self.in_row = []
        elif tag in ('th', 'td'):
            self.in_cell = []
            if tag == 'th':
                self.header_row = True
        elif tag in ('ul', 'ol'):
            self._flush()
            self.list_stack.append('-' if tag == 'ul' else '1.')
        elif tag == 'li':
            self.pending = []
        elif tag in ('dt', 'dd'):
            # glossary definition list - carries the plain-language explanations,
            # so it must survive conversion, not be silently dropped
            self._flush()
            self.pending = []
        elif tag == 'figcaption':
            self.fig_caption = True

    def handle_endtag(self, tag):
        if tag in ('style', 'script', 'title'):
            self.skip = max(0, self.skip - 1)
        elif tag in ('h1', 'h2', 'h3', 'h4'):
            self._flush(prefix=self._level)
        elif tag == 'p':
            self._flush()
        elif tag in ('th', 'td'):
            if self.in_row is not None and self.in_cell is not None:
                self.in_row.append(re.sub(r'\s+', ' ', ''.join(self.in_cell)).strip())
            self.in_cell = None
        elif tag == 'tr':
            if self.in_row:
                self._emit('| ' + ' | '.join(self.in_row) + ' |')
                if self.header_row:
                    self._emit('|' + '|'.join(['---'] * len(self.in_row)) + '|')
                    self.header_row = False
            self.in_row = None
        elif tag == 'table':
            self._emit('')
        elif tag == 'li':
            marker = self.list_stack[-1] if self.list_stack else '-'
            self._flush(prefix=marker + ' ')
        elif tag == 'dt':
            self._flush(prefix='**', suffix='**')
        elif tag == 'dd':
            self._flush(prefix='> ')
        elif tag in ('ul', 'ol'):
            if self.list_stack:
                self.list_stack.pop()
        elif tag == 'figcaption':
            self._flush(prefix='*', suffix='*')
            self.fig_caption = False

    def handle_data(self, data):
        if self.skip:
            return
        if self.in_cell is not None:
            self.in_cell.append(data)
        else:
            self.pending.append(data)

    def result(self):
        self._flush()
        text = '\n'.join(self.out)
        return re.sub(r'\n{3,}', '\n\n', text).strip() + '\n'


# --------------------------------------------------------------------- DOCX
def build_docx():
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        return None, ('python-docx is not installed, so docs/report.docx was not written.\n'
                      '  pip install python-docx     (then re-run)\n'
                      '  Meanwhile docs/report.md and docs/report.html are both importable '
                      'into Word directly.')

    s = read_template()
    for key, (fn, _) in FIGURES.items():
        s = s.replace('{{' + key + '}}', os.path.join(FIGS, FIGURES[key][0]))

    doc = Document()
    # Match the HTML report's typographic intent: serif body, monospace data.
    normal = doc.styles['Normal']
    normal.font.name = 'Georgia'
    normal.font.size = Pt(10.5)

    parser = _HtmlToDocx(doc)
    parser.feed(s)

    out = os.path.join(DOCS, 'report.docx')
    try:
        doc.save(out)
    except PermissionError:
        # Word (and some previewers) hold an exclusive lock on an open .docx.
        # Failing the whole build for that is unhelpful, so write alongside it
        # and tell the caller which file to use.
        alt = os.path.join(DOCS, 'report_new.docx')
        doc.save(alt)
        return alt, (f'{os.path.basename(out)} is locked by another program '
                     f'(close it in Word), so the new document was written to '
                     f'{os.path.basename(alt)} instead.')
    return out, None


class _HtmlToDocx(html.parser.HTMLParser):
    """Walks the report markup and writes the equivalent Word structure."""

    def __init__(self, doc):
        super().__init__(convert_charrefs=True)
        from docx.shared import Pt
        self.doc = doc
        self.Pt = Pt
        self.skip = 0
        self.buf = []
        self.mode = None          # 'h1'..'h4', 'p', 'li', 'caption'
        self.bold = 0
        self.rows = None
        self.row = None
        self.cell = None
        self.is_header = False
        self.tables = []
        self.list_kind = []

    def _text(self):
        t = re.sub(r'\s+', ' ', ''.join(self.buf)).strip()
        self.buf = []
        return t

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ('style', 'script', 'title'):
            self.skip += 1
        elif tag in ('h1', 'h2', 'h3', 'h4', 'p'):
            self.buf = []
            self.mode = tag
        elif tag in ('b', 'strong'):
            self.bold += 1
        elif tag == 'img':
            src = a.get('src', '')
            if os.path.exists(src):
                from docx.shared import Inches
                try:
                    self.doc.add_picture(src, width=Inches(6.2))
                except Exception:
                    pass
        elif tag == 'table':
            self.rows = []
        elif tag == 'tr':
            self.row = []
        elif tag in ('th', 'td'):
            self.cell = []
            if tag == 'th':
                self.is_header = True
        elif tag in ('ul', 'ol'):
            self.list_kind.append('List Bullet' if tag == 'ul' else 'List Number')
        elif tag == 'li':
            self.buf = []
            self.mode = 'li'
        elif tag in ('dt', 'dd'):
            self.buf = []
            self.mode = tag
        elif tag == 'figcaption':
            self.buf = []
            self.mode = 'caption'

    def handle_endtag(self, tag):
        if tag in ('style', 'script', 'title'):
            self.skip = max(0, self.skip - 1)
        elif tag in ('h1', 'h2', 'h3', 'h4'):
            t = self._text()
            if t:
                level = {'h1': 0, 'h2': 1, 'h3': 2, 'h4': 3}[tag]
                self.doc.add_heading(t, level=level)
            self.mode = None
        elif tag == 'p':
            t = self._text()
            if t:
                self.doc.add_paragraph(t)
            self.mode = None
        elif tag in ('b', 'strong'):
            self.bold = max(0, self.bold - 1)
        elif tag in ('th', 'td'):
            if self.row is not None and self.cell is not None:
                self.row.append(re.sub(r'\s+', ' ', ''.join(self.cell)).strip())
            self.cell = None
        elif tag == 'tr':
            if self.row:
                self.rows.append((self.is_header, self.row))
                self.is_header = False
            self.row = None
        elif tag == 'table':
            self._write_table()
            self.rows = None
        elif tag == 'li':
            t = self._text()
            if t:
                style = self.list_kind[-1] if self.list_kind else 'List Bullet'
                self.doc.add_paragraph(t, style=style)
            self.mode = None
        elif tag in ('ul', 'ol'):
            if self.list_kind:
                self.list_kind.pop()
        elif tag == 'dt':
            t = self._text()
            if t:
                par = self.doc.add_paragraph()
                par.add_run(t).bold = True
            self.mode = None
        elif tag == 'dd':
            t = self._text()
            if t:
                from docx.shared import Inches
                par = self.doc.add_paragraph(t)
                par.paragraph_format.left_indent = Inches(0.3)
            self.mode = None
        elif tag == 'figcaption':
            t = self._text()
            if t:
                p = self.doc.add_paragraph()
                r = p.add_run(t)
                r.italic = True
                r.font.size = self.Pt(9)
            self.mode = None

    def _write_table(self):
        if not self.rows:
            return
        ncols = max(len(r) for _, r in self.rows)
        table = self.doc.add_table(rows=0, cols=ncols)
        table.style = 'Light Grid Accent 1'
        for is_header, cells in self.rows:
            cells = cells + [''] * (ncols - len(cells))
            wr = table.add_row().cells
            for i, c in enumerate(cells):
                wr[i].text = c
                if is_header:
                    for par in wr[i].paragraphs:
                        for run in par.runs:
                            run.bold = True
        self.doc.add_paragraph()

    def handle_data(self, data):
        if self.skip:
            return
        if self.cell is not None:
            self.cell.append(data)
        elif self.mode:
            self.buf.append(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--formats', nargs='+', default=['html', 'md', 'docx'],
                    choices=['html', 'md', 'docx'])
    args = ap.parse_args()

    check_figures()
    print('building report from docs/report_template.html')

    if 'html' in args.formats:
        p = build_html()
        print(f'  [ok]   {os.path.relpath(p, ROOT)}  ({os.path.getsize(p)/1024/1024:.2f} MB, self-contained)')
    if 'md' in args.formats:
        p = build_markdown()
        print(f'  [ok]   {os.path.relpath(p, ROOT)}  ({os.path.getsize(p)/1024:.0f} KB, figures referenced)')
    if 'docx' in args.formats:
        p, warn = build_docx()
        if p:
            print(f'  [ok]   {os.path.relpath(p, ROOT)}  ({os.path.getsize(p)/1024:.0f} KB, Word)')
        else:
            print(f'  [skip] report.docx — {warn}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
