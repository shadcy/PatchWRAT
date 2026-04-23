import os
import glob

base_dir = r'c:\Users\Asus\Desktop\TTS\Rat\docs\report\plots-old'
latex = '\\newpage\n\\section*{Appendix B: Comprehensive Visualization Collection}\n\\label{app:comprehensive_plots}\n\n'
latex += 'The following gallery provides an exhaustive visual repository of the ablation studies, attention dynamics, spectral analysis, and filter evolutions spanning all forecasting horizons. \\textbf{All subsequent visualizations were generated explicitly w/pwsa.v8.0.}\n\n'

for root, dirs, files in os.walk(base_dir):
    for f in files:
        if f.endswith('.png'):
            rel_dir = os.path.relpath(root, base_dir).replace('\\', '/')
            if rel_dir == '.':
                rel_dir = ''
            else:
                rel_dir = rel_dir + '/'
            path = f'../report/plots-old/{rel_dir}{f}'
            
            clean_name = f.replace('_', ' ').replace('.png', '')
            caption = f'Visual representation of {clean_name}. Generated explicitly w/pwsa.v8.0.'
            
            latex += '\\begin{figure}[H]\n'
            latex += '    \\centering\n'
            latex += f'    \\includegraphics[width=0.8\\textwidth,height=0.35\\textheight,keepaspectratio]{{{path}}}\n'
            latex += f'    \\caption{{{caption}}}\n'
            latex += '\\end{figure}\n'
            latex += '\\clearpage\n\n'
            
with open(r'c:\Users\Asus\Desktop\TTS\Rat\docs\report2\appendix_b.tex', 'w') as out:
    out.write(latex)

print("Generated appendix_b.tex")
