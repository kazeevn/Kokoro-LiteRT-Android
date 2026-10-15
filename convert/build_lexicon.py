#!/usr/bin/env python3
"""Convert misaki's English lexicon JSONs into compact sorted TSV files for the Kotlin G2P.

Output (android/app/src/main/assets/g2p/): {us,gb}_{gold,silver}.tsv

Format: UTF-8, one entry per line, ``key<TAB>value\\n``, lines sorted by key in UTF-16
code-unit order (= Kotlin/Java String.compareTo order), so the app can binary-search the
file contents without building a hash map.

* Plain entries: value is the phoneme string.
* Heteronyms (dict-valued entries): value is ``{TAG=phonemes|TAG=phonemes...}``; a JSON
  null phoneme value is written as ``~``. TAG is the misaki key (DEFAULT, NOUN, VERB, ADJ,
  ADV, VBD, VBN, VBP, DT, None).

misaki's ``Lexicon.grow_dictionary`` (which roughly doubles the entry count by adding
Capitalized/lowercase variants) is NOT materialized, to halve the asset size; the Kotlin
``LexiconDict.get`` emulates it at lookup time. ``check_grow_emulation`` below verifies that
emulation against the real grow_dictionary for every key and its case variants.

Usage: python convert/build_lexicon.py [--data-dir DIR] [--out-dir DIR]
(by default the JSONs are taken from the installed misaki package).
"""
import argparse
import json
import os

PHONEME_FORBIDDEN = set('\t\n{}|=~')


def grow_dictionary(d):
    # Verbatim copy of misaki.en.Lexicon.grow_dictionary (misaki 0.9.4).
    e = {}
    for k, v in d.items():
        if len(k) < 2:
            continue
        if k == k.lower():
            if k != k.capitalize():
                e[k.capitalize()] = v
        elif k == k.lower().capitalize():
            e[k.lower()] = v
    return {**e, **d}


def capitalize(s):
    return s[:1].upper() + s[1:].lower()


def grown_get(d, w):
    """Lookup-time emulation of grow_dictionary(d).get(w) (mirrors LexiconDict.get in Kotlin)."""
    if w in d:
        return d[w]
    if len(w) < 2:
        return None
    lo = w.lower()
    cap = capitalize(lo)
    if w == lo and cap != lo:
        return d.get(cap) if cap in d else None
    if w == cap and w != lo:
        return d.get(lo)
    return None


def check_grow_emulation(d):
    g = grow_dictionary(d)
    probes = set(g)
    for k in d:
        probes.update((k.lower(), k.upper(), capitalize(k), k.capitalize()))
    for w in probes:
        assert grown_get(d, w) == g.get(w), (w, grown_get(d, w), g.get(w))


def encode_value(v):
    if isinstance(v, str):
        assert v and not (set(v) & PHONEME_FORBIDDEN), repr(v)
        return v
    assert isinstance(v, dict) and 'DEFAULT' in v, v
    parts = []
    for tag, ps in v.items():
        assert not (set(tag) & PHONEME_FORBIDDEN), tag
        if ps is None:
            ps = '~'
        else:
            assert ps and not (set(ps) & PHONEME_FORBIDDEN), repr(ps)
        parts.append(f'{tag}={ps}')
    return '{' + '|'.join(parts) + '}'


def utf16_key(s):
    return s.encode('utf-16-be')


def convert(src, dst):
    with open(src, encoding='utf-8') as f:
        d = json.load(f)
    check_grow_emulation(d)
    keys = sorted(d, key=utf16_key)
    with open(dst, 'w', encoding='utf-8', newline='\n') as f:
        for k in keys:
            # Kotlin compares keys byte-wise, so they must be printable ASCII.
            assert k and all(32 <= ord(c) < 127 for c in k), repr(k)
            f.write(f'{k}\t{encode_value(d[k])}\n')
    print(f'{os.path.basename(src)} -> {dst}: {len(keys)} entries, {os.path.getsize(dst)} bytes')


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=None, help='directory with misaki {us,gb}_{gold,silver}.json')
    ap.add_argument('--out-dir', default=os.path.join(root, 'android/app/src/main/assets/g2p'))
    args = ap.parse_args()
    data_dir = args.data_dir
    if data_dir is None:
        import misaki.data
        data_dir = os.path.dirname(misaki.data.__file__)
    os.makedirs(args.out_dir, exist_ok=True)
    for name in ('us_gold', 'us_silver', 'gb_gold', 'gb_silver'):
        convert(os.path.join(data_dir, name + '.json'), os.path.join(args.out_dir, name + '.tsv'))


if __name__ == '__main__':
    main()
