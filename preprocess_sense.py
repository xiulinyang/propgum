from glob import glob
from pathlib import Path
tabs = glob('tagger_new/*.tab')


for tab in tabs:
    tab_name = Path(tab).stem.split('.')[0]
    file_name = f'{tab_name}.sense.only.tab'
    with open(f'tagger_new/{file_name}', 'w') as f:
        tab_text = Path(tab).read_text().strip().split('\n\n')
        for text in tab_text:
            for l in text.split('\n'):
                info = l.split()
                if info[-1].startswith('B-s'):
                    f.write(f'{l}\n')
                else:
                    info[-1] ='O'
                    to_write = '\t'.join(info)
                    f.write(to_write + '\n')
            f.write('\n')




