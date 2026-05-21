# best snapshot

Created: 2026-05-17

Source: `/data/Maojie/Github2/EVRP-TW-D-B_Weekend`

Git branch at snapshot time: `dev`

Git HEAD at snapshot time: `d6eaba726f0c935e6d8252f66db4a8e6c2ee06aa`

Purpose: preserve the current best code state before further offline-injection / POMO50 optimization changes.

This snapshot was created with `rsync` from the working tree, so it includes uncommitted source/script changes that existed at snapshot time.

Excluded intentionally:

- `.git/`
- `dataset/`
- `dataset4/`
- `checkpoint/`
- `LOGS/`
- `LOGS_NEW/`
- `IMGS_NEW/`
- `imgs/`
- `version_snapshots/`
- Python cache directories and `.pyc` files

Restore pattern:

```bash
rsync -a version_snapshots/best/ ./
```

Use extra care if restoring into a dirty worktree; compare first if needed.
