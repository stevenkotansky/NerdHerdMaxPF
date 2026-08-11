# ESPN Max PF / Inverse Draft Order

Enter a **public** ESPN fantasy football league ID, a season, and a week. The
app pulls every team's full roster for that week (starters, bench, **and**
IR — all eligible), looks up what each player actually scored, and solves
for the mathematically optimal lineup each team *could* have started. That's
"Max PF." Teams are ranked by Max PF ascending, which is your inverse draft
order — the team with the lowest possible ceiling that week picks first.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Finding your league ID

It's in the URL when you're on your ESPN league's site, e.g.:
`https://fantasy.espn.com/football/league?leagueId=123456` → league ID is `123456`.

## Public vs. private leagues

This only works out of the box for **public** leagues. If your league is
private, ESPN's API needs your `espn_s2` and `SWID` cookies (grab them from
your browser's dev tools while logged into ESPN — Application/Storage →
Cookies → fantasy.espn.com). There's an optional field in the app for these;
they're sent straight to ESPN's API and never stored anywhere.

## How "Max PF" is computed

For each team, every rostered player (bench and IR included) is a candidate.
The league's actual starting lineup requirements (e.g. 1 QB, 2 RB, 2 WR, 1
TE, 1 FLEX, 1 D/ST, 1 K) are pulled from the league's own settings — this
isn't hardcoded, so it works whether your league runs standard, superflex,
2-QB, etc. The app then finds the assignment of players to slots that
maximizes total points, respecting each player's real position eligibility
(e.g. a TE can fill a FLEX slot but not a RB slot). This is solved exactly
as a bipartite max-weight matching problem (via `scipy.optimize.linear_sum_assignment`),
not a greedy approximation — so it's the true maximum possible score, not
just "close."

## Season Projection mode

Switch to the "Season Projection" tab for a live-updating projected end-of-season
Max PF standings — i.e. a running forecast of the inverse draft order:

- For every week **already played**, it uses that week's real roster and real
  actual points to compute that week's Max PF (same as Single Week mode),
  and sums them up.
- For every week **still to come**, it uses your team's *current* roster
  (today's roster — trades/waiver moves already reflected) and ESPN's own
  per-player weekly projections to compute a projected Max PF, and sums those.
- Total = completed-weeks Max PF + projected-remaining-weeks Max PF.

Two caveats worth knowing:
- **Roster assumption**: projections for future weeks assume your roster
  stays exactly as it is today. It won't guess future waiver moves or trades —
  re-run it anytime for an updated forecast.
- **"Final week" auto-detection**: the app guesses your regular season's last
  week from the league's playoff settings (team count + rounds). This is a
  best-effort guess — if your league has an unusual playoff structure, check
  the auto-detected week shown in the results and override it with the
  "Final week" field if it looks wrong.

## Notes

- "Actual PF" (shown alongside Max PF) is what the team's real starting
  lineup scored that week, for comparison.
- If your league has a unique lineup slot (e.g. a "Rookie" or superflex-style
  slot ESPN doesn't commonly expose), unmapped slots will show as `SLOTxx`
  in the lineup breakdown, but they're still calculated correctly.
