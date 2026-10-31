import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = 'results'
runs = [('AdamW', 'adamw', '#4C6EF5'), ('Muon (+aux AdamW)', 'muon', '#E8590C'), ('SOAP', 'soap', '#2F9E44'), ('Shampoo (Adam graft)', 'shampoo', '#9C36B5')]
fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150, sharex=True, sharey=True)
for ax, (name, f, c) in zip(axes.flat, runs):
    df = pd.read_csv(f'{R}/log_{f}.csv')
    ax.plot(df['step'], df['train_loss'], ls='--', marker='o', ms=3, lw=1.5, color=c, label='train')
    ax.plot(df['step'], df['val_loss'], ls='-', marker='o', ms=3, lw=1.8, color=c, label='val')
    b = df.loc[df['val_loss'].idxmin()]
    ax.axvline(b['step'], color='gray', lw=0.8, ls=':')
    ax.annotate(f"best val {b['val_loss']:.3f}\n@ step {int(b['step'])}", (b['step'], b['val_loss']), xytext=(8, 28), textcoords='offset points', fontsize=8)
    ax.set_title(name); ax.set_ylim(0.3, 2.6); ax.grid(alpha=.3); ax.legend(loc='upper right')
for ax in axes[1]: ax.set_xlabel('step')
for ax in axes[:, 0]: ax.set_ylabel('loss')
fig.suptitle('nanoGPT shakespeare-char: train (dashed) vs val (solid) loss', y=0.995)
fig.tight_layout(); fig.savefig(f'{R}/train_val_curves.png')
