# Build from this directory: latexmk
use Cwd qw(abs_path);
use File::Spec;
my $local_tex = "$ENV{HOME}/.local/share/kernel-schema-texlive/bin/universal-darwin";
$ENV{PATH} = "$local_tex:$ENV{PATH}" if -d $local_tex;
@default_files = ('main.tex');
$pdf_mode = 5; # XeLaTeX -> XDV -> PDF
$out_dir = 'build';
$aux_dir = 'build';
$recorder = 1;
$interaction = 'nonstopmode';
$xelatex = 'python3 scripts/compile_xelatex.py %O %S';
$success_cmd = 'python3 scripts/publish_pdf.py';
$clean_ext .= ' synctex.gz';
