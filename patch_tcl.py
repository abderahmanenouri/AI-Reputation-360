import json

notebook_path = r'c:\Users\abder\IdeaProjects\AI-Reputation-360\Analyse_Reputation_IA.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'source' in cell:
        source_code = "".join(cell['source'])
        if "df_tcl = df[df['Entreprise'] == 'TCL Lyon']" in source_code:
            cell['source'] = [line.replace("'TCL Lyon'", "'TCL_Lyon'") for line in cell['source']]
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('TCL name patched to TCL_Lyon in the notebook.')
