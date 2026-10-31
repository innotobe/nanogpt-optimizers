import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = 'results'
import os
all_runs = [('AdamW (2017)', 'adamw', '#4C6EF5'), ('Shampoo (2018, Adam graft)', 'shampoo', '#9C36B5'), ('SOAP (2024)', 'soap', '#2F9E44'), ('Muon (2024, +aux AdamW)', 'muon', '#E8590C')]
runs = {k: f'{R}/log_{f}.csv' for k, f, _ in all_runs if os.path.exists(f'{R}/log_{f}.csv')}
colors = {k: c for k, _, c in all_runs}
dfs = {k: pd.read_csv(v) for k, v in runs.items()}

for xcol, fname, xlabel in [('step', 'val_loss_vs_step.png', 'step'),
                            ('train_time_s', 'val_loss_vs_time.png', 'training wall-clock time (s, eval excluded)')]:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for k, df in dfs.items():
        ax.plot(df[xcol], df['val_loss'], marker='o', ms=3, lw=1.6, color=colors[k], label=k)
    ax.set_xlabel(xlabel); ax.set_ylabel('val loss'); ax.set_title(f'nanoGPT shakespeare-char: val loss vs {"step" if xcol == "step" else "wall-clock time"}')
    ax.set_ylim(1.4, 2.6); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(f'{R}/{fname}')

rows = []
for k, df in dfs.items():
    last = df.iloc[-1]; best = df.loc[df['val_loss'].idxmin()]
    n_train_steps = int(last['step']) - (len(df) - 1)  # eval-iteration steps are excluded from train_time
    rows.append(dict(run=k, final_val=last['val_loss'], best_val=best['val_loss'], best_step=int(best['step']),
                     final_train=last['train_loss'], train_time_s=last['train_time_s'], wall_time_s=last['wall_time_s'],
                     ms_per_step=1000 * last['train_time_s'] / n_train_steps))
tab = pd.DataFrame(rows)
tab.to_csv(f'{R}/summary_table.csv', index=False)
# steps/time to reach thresholds
thr = {}
for t in [2.0, 1.8, 1.7, 1.6, 1.5]:
    for k, df in dfs.items():
        hit = df[df['val_loss'] <= t]
        thr[(t, k)] = (int(hit.iloc[0]['step']), hit.iloc[0]['train_time_s']) if len(hit) else None
with open(f'{R}/summary_table.md', 'w') as f:
    f.write('| run | final val loss | best val loss (step) | final train loss | train time (s) | wall time incl. eval (s) | ms/step |\n|---|---|---|---|---|---|---|\n')
    for r in rows:
        f.write(f"| {r['run']} | {r['final_val']:.4f} | {r['best_val']:.4f} ({r['best_step']}) | {r['final_train']:.4f} | {r['train_time_s']:.1f} | {r['wall_time_s']:.1f} | {r['ms_per_step']:.1f} |\n")
    f.write('\n| val loss threshold | ' + ' | '.join(f'{k}: step (time s)' for k in dfs) + ' |\n|---|' + '---|' * len(dfs) + '\n')
    for t in [2.0, 1.8, 1.7, 1.6, 1.5]:
        f.write(f'| ≤ {t} | ' + ' | '.join((f'{v[0]} ({v[1]:.0f}s)' if (v := thr[(t, k)]) else 'not reached') for k in dfs) + ' |\n')
print(open(f'{R}/summary_table.md').read())
print(pd.concat([df.assign(run=k) for k, df in dfs.items()]).pivot(index='step', columns='run', values='val_loss').to_string())
