#!/usr/bin/env python3
"""Run the reference misaki 0.9.4 English G2P over a corpus and store expected outputs.

Used by the Kotlin port's JVM test (android/app/src/test/.../g2p/EnglishG2PCorpusTest.kt).

Usage: .venv-convert/bin/python convert/g2p_reference.py [corpus.txt out.jsonl]
Defaults: convert/g2p_test/{corpus.txt -> reference.jsonl, holdout.txt -> holdout_reference.jsonl,
holdout2.txt -> holdout2_reference.jsonl}
and convert/g2p_test/num2words.tsv (expected num2words outputs).

Each output line is a JSON object:
  text      the input sentence
  us, gb    misaki phoneme strings (G2P(trf=False, british=..., fallback=EspeakFallback))
  us_tokens, gb_tokens   [[text, whitespace, phonemes|null, tag], ...] misaki output tokens
  spacy     [[text, tag, whitespace], ...] spaCy en_core_web_sm tokens (for tokenizer debugging)
"""
import json
import os
import sys

from misaki import en, espeak

HERE = os.path.dirname(os.path.abspath(__file__))


def run_corpus(g2ps, corpus, out):
    nlp = g2ps['us'].nlp
    with open(corpus, encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]
    with open(out, 'w', encoding='utf-8') as f:
        for text in lines:
            rec = {'text': text}
            for k, g2p in g2ps.items():
                ps, tokens = g2p(text)
                rec[k] = ps
                rec[k + '_tokens'] = [[t.text, t.whitespace, t.phonemes, t.tag] for t in tokens]
            rec['spacy'] = [[t.text, t.tag_, t.whitespace_] for t in nlp(text)]
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'wrote {len(lines)} sentences to {out}')


def main():
    g2ps = {
        'us': en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False)),
        'gb': en.G2P(trf=False, british=True, fallback=espeak.EspeakFallback(british=True)),
    }
    test_dir = os.path.join(HERE, 'g2p_test')
    if len(sys.argv) > 2:
        run_corpus(g2ps, sys.argv[1], sys.argv[2])
    else:
        # corpus.txt and (after a first measurement) holdout.txt were used while developing the
        # Kotlin port's heuristics; holdout2.txt was not.
        run_corpus(g2ps, os.path.join(test_dir, 'corpus.txt'), os.path.join(test_dir, 'reference.jsonl'))
        run_corpus(g2ps, os.path.join(test_dir, 'holdout.txt'), os.path.join(test_dir, 'holdout_reference.jsonl'))
        run_corpus(g2ps, os.path.join(test_dir, 'holdout2.txt'), os.path.join(test_dir, 'holdout2_reference.jsonl'))
    write_num2words(os.path.join(test_dir, 'num2words.tsv'))


def write_num2words(path):
    """Expected num2words outputs (kind, input, output) for the Kotlin Num2Words unit test."""
    from num2words import num2words
    nums = list(range(0, 2101)) + [10**k + d for k in range(3, 22) for d in (0, 1, 7, 99, 101, 999, 12345)]
    nums += [123456789, 987654321012, 1000000000000000000000000, 31415926535897932384]
    rows = []
    for n in nums:
        rows.append(('cardinal', str(n), num2words(n)))
        rows.append(('ordinal', str(n), num2words(n, to='ordinal')))
    for n in range(1000, 10000, 7):
        rows.append(('year', str(n), num2words(n, to='year')))
    for s in ['3.14', '3.14159', '0.5', '0.50', '2.0', '100.50', '0.25', '7.8', '98.6', '1.10', '12.345',
              '0.001', '1000.01', '3.14159265358979323', '42.000', '0.07', '19.99', '1.5', '2.25']:
        rows.append(('float', s, num2words(float(s))))
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write('\t'.join(r) + '\n')
    print(f'wrote {len(rows)} num2words cases to {path}')


if __name__ == '__main__':
    main()
