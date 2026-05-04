# --- output directory ---
$out_dir = 'build';

# --- emulate aux separation ---
$emulate_aux = 1;

# --- output PDF name ---
$jobname = 'report';

# --- compiler ---
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# --- bibliography ---
$bibtex = 'bibtex %O %S';

# --- move PDF to root (no duplication) ---
END {
    if (-e 'build/report.pdf') {
        rename 'build/report.pdf', 'report.pdf';
    }
}

# create build folder
if (!-d 'build') { mkdir 'build'; }