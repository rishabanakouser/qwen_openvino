# fonts/

Place TrueType (.ttf) fonts here for the text renderer.

## Required font

The default configuration expects:

```
fonts/DejaVuSans.ttf
```

## Download DejaVuSans

### Option A — wget / curl

```bash
# Linux / macOS
wget -O fonts/DejaVuSans.ttf \
  "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.tar.bz2"
# Then extract DejaVuSans.ttf from the archive.
```

### Option B — pip / system package (Linux)

```bash
sudo apt-get install fonts-dejavu
cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf ./fonts/
```

### Option C — Manual download

1. Visit https://dejavu-fonts.github.io/
2. Download `dejavu-fonts-ttf-*.tar.bz2`
3. Extract and copy `DejaVuSans.ttf` to this folder.

### Option D — Use the provided script

```bash
python download_font.py
```

## Using a different font

Set `FONT_PATH` in `.env`:

```
FONT_PATH=./fonts/NotoSans-Regular.ttf
```

Noto fonts work well for multilingual text including CJK characters:
https://fonts.google.com/noto

## Fallback

If no font file is found, the service falls back to Pillow's built-in
bitmap font. Text will still be rendered but at a fixed small size
without Unicode support.
