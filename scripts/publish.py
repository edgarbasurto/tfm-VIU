import argparse, os, shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--overleaf-dir', required=True, help='Carpeta de tu proyecto Overleaf sincronizado')
    ap.add_argument('--outputs-dir', default='outputs')
    args = ap.parse_args()

    figs_src = os.path.join(args.outputs_dir, 'figures')
    tabs_src = os.path.join(args.outputs_dir, 'tables')

    figs_dst = os.path.join(args.overleaf_dir, 'figures')
    tabs_dst = os.path.join(args.overleaf_dir, 'tables')

    os.makedirs(figs_dst, exist_ok=True)
    os.makedirs(tabs_dst, exist_ok=True)

    # copiar PDFs/PNGs
    for f in os.listdir(figs_src):
        shutil.copy2(os.path.join(figs_src, f), os.path.join(figs_dst, f))
    # copiar .tex
    for f in os.listdir(tabs_src):
        shutil.copy2(os.path.join(tabs_src, f), os.path.join(tabs_dst, f))

    print('Publicación a Overleaf completada.')

if __name__ == '__main__':
    main()
