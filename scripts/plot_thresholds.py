"""Plot the retrospective recall vs alarm points from recomputed CSV.

This is a descriptive plot of post-NMS, score-censored saved predictions.
It does not represent independently calibrated operating points.
"""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(a.csv.open(newline='', encoding='utf8')))
    models = dict()
    for row in rows:
        models.setdefault(row['model'], []).append(row)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model, data in models.items():
        pts = sorted(data, key=lambda x: float(x['threshold']))
        ax.plot([100 * float(x['recall']) for x in pts],
                [100 * float(x['good_alarm_rate']) for x in pts],
                marker='o', lw=1.5, label=model.replace('_', ' '))
    ax.set_xlabel('Annotated-object recall (%)')
    ax.set_ylabel('Alarm rate on dataset-labelled good images (%)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=220)
    print('Wrote', a.out)

if __name__ == '__main__':
    main()
