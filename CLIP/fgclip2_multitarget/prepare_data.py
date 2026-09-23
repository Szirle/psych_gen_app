"""Bootstrap public OMI or transfer the existing trusted-local dataset.

This helper deliberately does not import PyTorch or unpickle downloaded data.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import pickle
import shutil
import tarfile
import zipfile


OMI_REVISION = '53bb3c13113afc3ae67aa6ede4ac9dfe33f306af'


def unpack(archive, output, omi=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:*') as tar:
        for member in tar:
            parts = Path(member.name).parts
            if '..' in parts or Path(member.name).is_absolute():
                raise ValueError('Unsafe archive member path')
            if not member.isfile():
                continue
            # Copy only direct images and explicitly recognized data files;
            # links, executable files, nested paths and validation_images ignored.
            if len(parts) >= 2 and parts[-2] == 'images':
                if Path(parts[-1]).suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
                    continue
                destination = output/'images'/parts[-1]
            elif parts[-1] in (('attribute_ratings.zip', 'LICENSE.txt') if omi else ('ratings.pkl',)):
                destination = output/parts[-1]
            else:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                continue
            temporary = destination.with_suffix(destination.suffix+'.tmp')
            with tar.extractfile(member) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target)
            temporary.replace(destination)
    if omi:
        mapping = {}
        images = {p.stem: p.name for p in (output/'images').iterdir() if p.is_file()}
        if len(images) != 1004:
            raise ValueError(f'Pinned OMI should have 1,004 images; found {len(images)}')
        with zipfile.ZipFile(output/'attribute_ratings.zip') as z:
            with z.open('attribute_ratings.csv') as source:
                for row in csv.DictReader(io.TextIOWrapper(source)):
                    filename = images.get(row['stimulus'])
                    if filename is None:
                        continue
                    if row['rating_type'] not in ('normal', 'repeat', 'fill-in'):
                        raise ValueError(f'Unrecognized rating type: {row["rating_type"]}')
                    rating = float(row['rating'])
                    if not 0 <= rating <= 100:
                        raise ValueError('Unexpected raw OMI rating scale')
                    # Match this project's existing slider conversion, including
                    # repeat/fill-in ratings: public integer r -> (r + .5)/101.
                    mapping.setdefault(row['attribute'], {}).setdefault(filename, []).append((rating+.5)/101)
        temporary = output/'ratings.pkl.tmp'
        with temporary.open('wb') as f:
            pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(output/'ratings.pkl')
        (output/'provenance.json').write_text(json.dumps(dict(source='https://github.com/jcpeterson/omi',
            revision=OMI_REVISION, scale='(public_rating + 0.5) / 101', rating_types=['normal', 'repeat', 'fill-in'],
            dimensions=sorted(mapping)), indent=2)+'\n')
    if not (output/'ratings.pkl').is_file() or not (output/'images').is_dir():
        raise ValueError('Dataset archive must contain images/ and ratings.pkl')
    print(f'Dataset ready: {output.resolve()}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('omi', 'unpack', 'pack'))
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--ratings', type=Path)
    args = parser.parse_args()
    if args.action == 'pack':
        if not args.images or not args.ratings:
            parser.error('pack requires --images and --ratings')
        if args.archive.exists():
            raise FileExistsError(args.archive)
        with tarfile.open(args.archive, 'w:gz') as tar:
            tar.add(args.ratings, arcname='ratings.pkl')
            for path in sorted(args.images.iterdir()):
                if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
                    tar.add(path, arcname=f'images/{path.name}', recursive=False)
    else:
        if not args.output:
            parser.error('unpack/omi requires --output')
        unpack(args.archive, args.output, omi=args.action == 'omi')


if __name__ == '__main__':
    main()
