# Cake live Podly deploy (`podly.service`)

Live checkout is `/home/jdr/cursor/repos/podly_pure_podcasts` (systemd
`WorkingDirectory`). The experiment branch is checked out in the scorecard
worktree `/home/jdr/cursor/repos/podly_scorecard`, so the live dir uses a
detached commit (git refuses the same branch in two worktrees).

Current live (2026-10-10): detached `cursor/algo-theories-2026-10-10` with
`ENABLE_QUOTE_ANCHOR_BOUNDARIES=true`, `ENABLE_SPOT_THEN_CONFIRM=true`,
`ENABLE_SNAP_EARLIER_START=true`, `ENABLE_SPOT_WEAK_DROP=true`. See
`ALGO_THEORIES_2026-10-10.md`. Increment rollback: `git checkout --detach
1b7728d` (keep quote-anchor + spot-then-confirm; unset the two new flags).
Spot-then-confirm off: `git checkout --detach e902c89` and unset
`ENABLE_SPOT_THEN_CONFIRM`. Quote-anchor off: unset those flags and
`git checkout --detach 20ed084`.

`enable_bow_scout_gemini_confirm` stays **false** (code default; not in
`.env.local`). Do not set it.

## Recorded rollback identity (2026-10-05 09:15 PT)

| item | value |
| --- | --- |
| live checkout | `/home/jdr/cursor/repos/podly_pure_podcasts` |
| live branch before deploy | `main` |
| live commit before deploy | `0b8160ac7770593d3c99f3c7e525e5882859be4e` |
| experiment | `cursor/opening-fn-2026-10-05` in `/home/jdr/cursor/repos/podly_scorecard` (first live switch `5f9dc22`; precision fold `180d749`+) |
| instance dir | `/home/jdr/cursor/repos/podly_pure_podcasts/src/instance` |
| timestamped backup | `/home/jdr/cursor/tmp/podly_deploy_backup_20261005_091522` |

Backup contains `instance/sqlite3.db` (Python `sqlite3.backup`, consistent),
`instance/jobs.sqlite`, `config/config.yml`, `.env.local`. Do not print those
files.

## Rollback (exact)

Restore the pre-experiment commit and restart both units. Instance DB/config
are gitignored and stay put unless you also restore the backup.

```bash
LIVE=/home/jdr/cursor/repos/podly_pure_podcasts
BACKUP=/home/jdr/cursor/tmp/podly_deploy_backup_20261005_091522
PREV=0b8160ac7770593d3c99f3c7e525e5882859be4e

systemctl --user stop podly.service
git -C "$LIVE" checkout --detach "$PREV"
systemctl --user restart podly-writer.service
sleep 3
systemctl --user start podly.service
systemctl --user is-active podly.service podly-writer.service
curl -sf -o /dev/null http://127.0.0.1:5001/ && echo web_ok
```

Equivalent if no other worktree holds `main`: `git -C "$LIVE" checkout main`
(that was `0b8160a` at deploy time).

Restore DB/config only if the experiment migrated or mutated them:

```bash
LIVE=/home/jdr/cursor/repos/podly_pure_podcasts
BACKUP=/home/jdr/cursor/tmp/podly_deploy_backup_20261005_091522

systemctl --user stop podly.service podly-writer.service
cp -a "$BACKUP/instance/sqlite3.db" "$LIVE/src/instance/sqlite3.db"
rm -f "$LIVE/src/instance/sqlite3.db-wal" "$LIVE/src/instance/sqlite3.db-shm"
cp -a "$BACKUP/instance/jobs.sqlite" "$LIVE/src/instance/jobs.sqlite"
cp -a "$BACKUP/config/config.yml" "$LIVE/config/config.yml"
cp -a "$BACKUP/.env.local" "$LIVE/.env.local"
systemctl --user start podly-writer.service
sleep 3
systemctl --user start podly.service
```

## Forward deploy (detached experiment commit)

```bash
LIVE=/home/jdr/cursor/repos/podly_pure_podcasts
EXP=$(git -C /home/jdr/cursor/repos/podly_scorecard rev-parse cursor/opening-fn-2026-10-05)

# Do not `uv sync` the live `.venv` (it tries to pull CUDA torch). Lock was
# identical; flask-migrate runs on writer start (`upgrade(revision="heads")`).
systemctl --user stop podly.service
git -C "$LIVE" fetch --all --tags  # optional; commit is already local
git -C "$LIVE" checkout --detach "$EXP"
systemctl --user restart podly-writer.service
sleep 3
systemctl --user start podly.service
```

After a later commit on `cursor/opening-fn-2026-10-05`, replace `$EXP` with
that SHA (still detach; do not check the branch out in `$LIVE` while the
scorecard worktree holds it).

## Health check

```bash
systemctl --user is-active podly.service podly-writer.service
curl -sf -o /dev/null http://127.0.0.1:5001/ && echo local_ui_ok
curl -sf http://127.0.0.1:5001/feeds | python3 -c "import json,sys; titles=[f['title'] for f in json.load(sys.stdin)]; print(titles)"
curl -sf -o /dev/null http://podly.jdrbc.ca:5001/ && echo pub_ok
journalctl --user -u podly.service -u podly-writer.service --since "2 min ago" --no-pager
```

Expected feeds: Patrick Boyle On Finance, Prof G Markets, Planet Money, The
Indicator from Planet Money. Scout flag must stay false.
