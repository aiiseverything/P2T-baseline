#!/usr/bin/env python3
"""Reproduce the four-case paper heatmap using only the bundled saved JSON."""
import argparse
import json
import os
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplconfig'))
from matplotlib.font_manager import fontManager
import render_credit_case_study_original as renderer

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE/'reproduced')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for font in (HERE/'fonts').glob('*.ttf'):fontManager.addfont(str(font))
    source=HERE/(renderer.STEM+'.json');data=json.loads(source.read_text())
    for case in data['cases']:
        for token in case['tokens']:
            assert .25<=token['weight']<=4
            assert abs(token['signed_advantage']-case['advantage']*token['weight'])<1e-9
    layout=renderer.render(data['cases'],4,a.out,data['rendering']['gamma'],data['rendering']['png_dpi'])
    assert len(layout['token_boxes'])==76
    for suffix in ('.json','_tokens.csv','.md'):
        path=HERE/(renderer.STEM+suffix)
        if path.resolve()!=(a.out/path.name).resolve():shutil.copy2(path,a.out/path.name)
    print('Reproduced four saved cases and76 original token weights; no model or training files required.')
