
import os

base_path = r"../report/plots-old"
groups = {
    "Attention Analysis": "attention",
    "Benchmarking Benchmarks": "benchmarks",
    "Filter Dynamics": "filters",
    "Learning Trajectories": "learning_curves",
    "Graph Topological Analysis": "variate_graph"
}

horizons = ["H96", "H192", "H336", "H720"]

latex = [
    "\\clearpage",
    "\\section*{Appendix B: Supplementary Performance Visualizations (w/pwsa.v8.0)}",
    "\\label{app:plots_old}",
    "",
    "This appendix contains the complete set of visualization results generated during the development of \\texttt{pwsa.v8.0}. These plots validate the spectral decomposition, attention routing, and forecasting fidelity of the architecture across all tested horizons.",
    ""
]

# Simple groups
for title, folder in groups.items():
    latex.append(f"\\subsection*{{B.{list(groups.keys()).index(title)+1} {title} (w/pwsa.v8.0)}}")
    files = sorted(os.listdir(os.path.join(r"c:\Users\Asus\Desktop\TTS\Rat\docs\report\plots-old", folder)))
    
    # Group by pairs for side-by-side
    for i in range(0, len(files), 2):
        latex.append("\\begin{figure}[H]")
        latex.append("    \\centering")
        
        f1 = files[i]
        p1 = f"{base_path}/{folder}/{f1}".replace("\\", "/")
        caption1 = f"{f1.replace('_', ' ')} w/pwsa.v8.0"
        
        if i + 1 < len(files):
            f2 = files[i+1]
            p2 = f"{base_path}/{folder}/{f2}".replace("\\", "/")
            caption2 = f"{f2.replace('_', ' ')} w/pwsa.v8.0"
            
            latex.append(f"    \\begin{{subfigure}}[b]{{0.48\\textwidth}}")
            latex.append(f"        \\centering")
            latex.append(f"        \\includegraphics[width=\\textwidth]{{{p1}}}")
            latex.append(f"        \\caption{{{caption1}}}")
            latex.append(f"    \\end{{subfigure}}")
            latex.append("    \\hfill")
            latex.append(f"    \\begin{{subfigure}}[b]{{0.48\\textwidth}}")
            latex.append(f"        \\centering")
            latex.append(f"        \\includegraphics[width=\\textwidth]{{{p2}}}")
            latex.append(f"        \\caption{{{caption2}}}")
            latex.append(f"    \\end{{subfigure}}")
        else:
            latex.append(f"    \\includegraphics[width=0.7\\textwidth]{{{p1}}}")
            latex.append(f"    \\caption{{{caption1}}}")
            
        latex.append("\\end{figure}")
    latex.append("\\clearpage")

# Predictions
latex.append("\\subsection*{B.6 Comprehensive Multivariate Predictions (w/pwsa.v8.0)}")
for h in horizons:
    latex.append(f"\\subsubsection*{{Predictions for {h} (w/pwsa.v8.0)}}")
    pred_dir = os.path.join(r"c:\Users\Asus\Desktop\TTS\Rat\docs\report\plots-old\predictions", h)
    if not os.path.exists(pred_dir): continue
    
    files = sorted(os.listdir(pred_dir))
    
    # Place special files first
    special = ["all_variates_grid.png", "horizon_error_profile.png", "residuals.png"]
    for s in special:
        if s in files:
            latex.append("\\begin{figure}[H]")
            latex.append("    \\centering")
            p = f"{base_path}/predictions/{h}/{s}".replace("\\", "/")
            latex.append(f"    \\includegraphics[width=\\textwidth]{{{p}}}")
            latex.append(f"    \\caption{{{h} {s.replace('_', ' ')} w/pwsa.v8.0}}")
            latex.append("\\end{figure}")
            files.remove(s)
    
    # Group variate samples (usually 3 per variate)
    # variate_0_HUFL_sample0.png, variate_0_HUFL_sample1.png, variate_0_HUFL_sample2.png
    variates = {}
    for f in files:
        if f.startswith("variate_"):
            parts = f.split("_")
            v_key = "_".join(parts[:3]) # e.g. variate_0_HUFL
            if v_key not in variates: variates[v_key] = []
            variates[v_key].append(f)
    
    for v_key, v_files in variates.items():
        latex.append("\\begin{figure}[H]")
        latex.append("    \\centering")
        for i, val in enumerate(v_files):
            p = f"{base_path}/predictions/{h}/{val}".replace("\\", "/")
            cap = f"{val.replace('_', ' ')} w/pwsa.v8.0"
            latex.append(f"    \\begin{{subfigure}}[b]{{0.32\\textwidth}}")
            latex.append(f"        \\centering")
            latex.append(f"        \\includegraphics[width=\\textwidth]{{{p}}}")
            latex.append(f"        \\caption{{{cap}}}")
            latex.append(f"    \\end{{subfigure}}")
            if (i+1) % 3 == 0 and i < len(v_files)-1:
                latex.append("\\\\") # New row in figure
        latex.append(f"    \\caption{{Multivariate predictions for {v_key} at horizon {h} w/pwsa.v8.0}}")
        latex.append("\\end{figure}")
    
    latex.append("\\clearpage")

print("\n".join(latex))
