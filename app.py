"""
ESPN Fantasy Football - Max Possible Points (Max PF) Calculator
=================================================================

Given a public ESPN fantasy football league ID + season + week, this app:
  1. Pulls every team's full roster (starters, bench, AND injured reserve)
     for that week.
  2. Pulls each rostered player's actual points scored that week.
  3. Computes the mathematically optimal lineup for each team that week
     (i.e. "if you had started your best possible lineup from everyone
     on your roster, bench and IR included, what's the max you could
     have scored?").
  4. Ranks teams by Max PF ascending -> that's your inverse draft order
     (worst possible-week first).

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

from flask import Flask, render_template_string, request, jsonify
import requests
from scipy.optimize import linear_sum_assignment

app = Flask(__name__)

ESPN_BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{year}/segments/0/leagues/{league_id}"

# Standard ESPN fantasy football lineup slot ID -> label.
# (Community-documented mapping; covers the slots that show up in >99% of leagues.)
SLOT_MAP = {
    0: "QB", 1: "TQB", 2: "RB", 3: "RB/WR", 4: "WR", 5: "WR/TE", 6: "TE",
    7: "OP", 8: "DT", 9: "DE", 10: "LB", 11: "DL", 12: "CB", 13: "S",
    14: "DB", 15: "DP", 16: "D/ST", 17: "K", 18: "P", 19: "HC",
    20: "BE", 21: "IR", 22: "UNKNOWN", 23: "FLEX", 24: "EDR", 25: "KOP",
}
BENCH_SLOTS = {20, 21, 22, 24, 25}  # slots that don't count toward a starting lineup

# --- League sidebar: current champion + marriage tracker ---
CURRENT_CHAMPION = "Steven Kotansky"
MARRIAGE_STATUS = {
    "justin shaw": ("Justin Shaw", "First Married"),
    "steven kotansky": ("Steven Kotansky", "Second Married"),
    "jackson selby": ("Jackson Selby", "Engaged"),
}


# ESPN player "position id" -> label, used as a fallback for eligibleSlots
PRO_POS_MAP = {
    0: "QB", 1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST",
}


def espn_get(url, params, espn_s2=None, swid=None):
    cookies = {}
    if espn_s2 and swid:
        cookies = {"espn_s2": espn_s2, "SWID": swid}
    resp = requests.get(url, params=params, cookies=cookies, timeout=20)
    if resp.status_code == 401:
        raise PermissionError(
            "This league is private (ESPN returned 401). Public leagues only are "
            "supported here — make the league public in ESPN's site settings, or "
            "this tool would need your espn_s2/SWID cookies added."
        )
    if resp.status_code == 404:
        raise LookupError("League/season not found. Check the league ID and year.")
    resp.raise_for_status()
    return resp.json()


def fetch_league(league_id, year, week, espn_s2=None, swid=None):
    """week=None fetches the league's 'current' snapshot (no scoringPeriodId
    filter) -- current roster + status, with each player's full stats blob
    (past actuals and future projections alike)."""
    url = ESPN_BASE.format(year=year, league_id=league_id)
    params = [
        ("view", "mRoster"),
        ("view", "mTeam"),
        ("view", "mSettings"),
        ("view", "mMatchupScore"),
    ]
    if week is not None:
        params.append(("scoringPeriodId", week))
    return espn_get(url, params, espn_s2, swid)


def get_lineup_slot_counts(data):
    """Return {slotId: count} of starting (non-bench) slots from league settings."""
    settings = data.get("settings", {})
    roster_settings = settings.get("rosterSettings", {})
    counts = roster_settings.get("lineupSlotCounts", {})
    starting = {}
    for slot_id_str, count in counts.items():
        slot_id = int(slot_id_str)
        if slot_id in BENCH_SLOTS:
            continue
        if count and count > 0:
            starting[slot_id] = count
    return starting


def player_week_points(player, week, source_id=0):
    """
    Points for the given scoring period.
    source_id=0 -> actual, source_id=1 -> ESPN's own projection.
    Player stat blobs returned by ESPN include entries for many weeks at once
    (past actual + future projected), so this just looks up the right one.
    """
    for stat in player.get("stats", []):
        if stat.get("scoringPeriodId") == week and stat.get("statSourceId") == source_id:
            return float(stat.get("appliedTotal", 0.0) or 0.0)
    return 0.0


def estimate_regular_season_final_week(data):
    """
    Best-effort read of the league's regular-season length from its own settings
    (total matchup periods minus estimated playoff weeks). Returns None if the
    league's settings don't expose enough to guess confidently -- callers should
    treat this as an editable default, not gospel.
    """
    sched = data.get("settings", {}).get("scheduleSettings", {})
    matchup_period_count = sched.get("matchupPeriodCount")
    if not matchup_period_count:
        return None
    playoff_team_count = sched.get("playoffTeamCount") or 0
    playoff_matchup_length = sched.get("playoffMatchupPeriodLength") or 1
    if not playoff_team_count:
        return matchup_period_count
    import math
    playoff_rounds = max(1, math.ceil(math.log2(playoff_team_count)))
    playoff_weeks = playoff_rounds * playoff_matchup_length
    return max(matchup_period_count - playoff_weeks, 1)


def get_current_week(data):
    return (
        data.get("status", {}).get("currentMatchupPeriod")
        or data.get("scoringPeriodId")
        or 1
    )


def get_member_names(data):
    """League members' display names, from the league response's 'members' list."""
    names = []
    for m in data.get("members", []):
        name = (m.get("displayName") or "").strip()
        if not name:
            name = f"{m.get('firstName', '')} {m.get('lastName', '')}".strip()
        if name:
            names.append(name)
    return names


def build_sidebar(data):
    """Current champion + marriage tracker, with 'other' names pulled live
    from this league's member list (so it's not hardcoded to one league)."""
    member_names = get_member_names(data)

    tracker = {"First Married": None, "Second Married": None, "Engaged": None}
    matched_keys = set()
    others = []

    for name in member_names:
        key = name.strip().lower()
        if key in MARRIAGE_STATUS:
            canonical_name, status = MARRIAGE_STATUS[key]
            tracker[status] = canonical_name
            matched_keys.add(key)
        else:
            others.append(name)

    # Include any of the three special names even if they weren't found among
    # this league's members (e.g. testing against a league they're not in).
    for key, (canonical_name, status) in MARRIAGE_STATUS.items():
        if key not in matched_keys and tracker[status] is None:
            tracker[status] = canonical_name

    return {
        "current_champion": CURRENT_CHAMPION,
        "marriage_tracker": {
            "first_married": tracker["First Married"],
            "second_married": tracker["Second Married"],
            "engaged": tracker["Engaged"],
            "other": others,
        },
    }


def team_display_name(team):
    name = team.get("name")
    if name:
        return name
    loc = team.get("location", "").strip()
    nick = team.get("nickname", "").strip()
    combined = (loc + " " + nick).strip()
    return combined or f"Team {team.get('id')}"


def compute_max_pf(entries, slot_counts, week):
    """
    entries: list of dicts {name, points, eligible_slots(set of slot ids)}
    slot_counts: {slotId: count} of starting slots to fill
    Returns (max_pf, chosen_lineup [{name, slot, points}], actual_starters_pf)
    """
    # Expand slot_counts into individual slot instances, e.g. RB:2 -> [2, 2]
    slot_instances = []
    for slot_id, count in slot_counts.items():
        slot_instances.extend([slot_id] * count)

    n_players = len(entries)
    n_slots = len(slot_instances)
    if n_players == 0 or n_slots == 0:
        return 0.0, []

    INFEASIBLE = 1_000_000.0
    # cost[i][j] = -points if player i eligible for slot j else INFEASIBLE
    cost = []
    for e in entries:
        row = []
        for slot_id in slot_instances:
            if slot_id in e["eligible_slots"]:
                row.append(-e["points"])
            else:
                row.append(INFEASIBLE)
        cost.append(row)

    row_ind, col_ind = linear_sum_assignment(cost)

    lineup = []
    total = 0.0
    for r, c in zip(row_ind, col_ind):
        if cost[r][c] >= INFEASIBLE:
            continue  # slot went unfilled (not enough eligible players)
        e = entries[r]
        slot_label = SLOT_MAP.get(slot_instances[c], f"SLOT{slot_instances[c]}")
        lineup.append({"name": e["name"], "slot": slot_label, "points": round(e["points"], 2)})
        total += e["points"]

    lineup.sort(key=lambda x: -x["points"])
    return round(total, 2), lineup


def compute_league(league_id, year, week, espn_s2=None, swid=None):
    data = fetch_league(league_id, year, week, espn_s2, swid)
    slot_counts = get_lineup_slot_counts(data)
    if not slot_counts:
        raise ValueError("Could not read starting lineup slots from league settings.")

    teams = data.get("teams", [])
    results = []

    for team in teams:
        roster = team.get("roster", {})
        raw_entries = roster.get("entries", [])

        entries = _roster_entries(team, week, source_id=0)

        actual_starters_pf = 0.0
        for re, e in zip(raw_entries, entries):
            current_slot = re.get("lineupSlotId")
            if current_slot is not None and current_slot not in BENCH_SLOTS:
                actual_starters_pf += e["points"]

        max_pf, lineup = compute_max_pf(entries, slot_counts, week)

        results.append({
            "team": team_display_name(team),
            "team_id": team.get("id"),
            "actual_pf": round(actual_starters_pf, 2),
            "max_pf": max_pf,
            "points_left_on_bench": round(max_pf - actual_starters_pf, 2),
            "optimal_lineup": lineup,
            "roster_size": len(entries),
        })

    # Inverse draft order: lowest Max PF picks first.
    results.sort(key=lambda r: r["max_pf"])
    for i, r in enumerate(results, start=1):
        r["inverse_draft_pick"] = i

    return {
        "league_id": league_id,
        "year": year,
        "week": week,
        "league_name": data.get("settings", {}).get("name", ""),
        "results": results,
        "sidebar": build_sidebar(data),
    }


def compute_season_projection(league_id, year, espn_s2=None, swid=None, final_week_override=None):
    """
    Projected end-of-season Max PF standings:
      - For every already-completed week: each team's ACTUAL best-possible
        lineup that week (roster + points as they really were that week).
      - For every remaining week: the CURRENT roster's best-possible lineup
        using ESPN's own per-week projections.
    Summed together, sorted ascending -> a live-updating projected inverse
    draft order.
    """
    # Base fetch (no scoringPeriodId -> "current" snapshot): gives us the
    # current roster, league settings, and where we are in the season.
    base = fetch_league(league_id, year, week=None, espn_s2=espn_s2, swid=swid)

    slot_counts = get_lineup_slot_counts(base)
    if not slot_counts:
        raise ValueError("Could not read starting lineup slots from league settings.")

    current_week = get_current_week(base)
    final_week = final_week_override or estimate_regular_season_final_week(base)
    if not final_week:
        raise ValueError(
            "Couldn't auto-detect this league's regular-season length. "
            "Set 'Final week' manually and try again."
        )
    if final_week < 1:
        raise ValueError("Final week must be 1 or greater.")

    teams = base.get("teams", [])
    team_names = {t.get("id"): team_display_name(t) for t in teams}
    totals = {tid: {"completed_max_pf": 0.0, "projected_remaining_max_pf": 0.0} for tid in team_names}

    completed_weeks = [w for w in range(1, current_week) if w <= final_week]
    remaining_weeks = [w for w in range(current_week, final_week + 1)]

    weeks_detail = {tid: [] for tid in team_names}

    # --- Completed weeks: real roster-as-it-was + real actual points ---
    for wk in completed_weeks:
        wk_data = fetch_league(league_id, year, wk, espn_s2, swid)
        for team in wk_data.get("teams", []):
            tid = team.get("id")
            if tid not in totals:
                continue
            entries = _roster_entries(team, wk, source_id=0)
            max_pf, _ = compute_max_pf(entries, slot_counts, wk)
            totals[tid]["completed_max_pf"] += max_pf
            weeks_detail[tid].append({"week": wk, "type": "actual", "max_pf": max_pf})

    # --- Remaining weeks: current roster + ESPN's projections ---
    for wk in remaining_weeks:
        for team in teams:
            tid = team.get("id")
            if tid not in totals:
                continue
            entries = _roster_entries(team, wk, source_id=1)
            max_pf, _ = compute_max_pf(entries, slot_counts, wk)
            totals[tid]["projected_remaining_max_pf"] += max_pf
            weeks_detail[tid].append({"week": wk, "type": "projected", "max_pf": max_pf})

    results = []
    for tid, name in team_names.items():
        completed = round(totals[tid]["completed_max_pf"], 2)
        remaining = round(totals[tid]["projected_remaining_max_pf"], 2)
        results.append({
            "team": name,
            "team_id": tid,
            "completed_weeks": len(completed_weeks),
            "remaining_weeks": len(remaining_weeks),
            "cumulative_max_pf_to_date": completed,
            "projected_remaining_max_pf": remaining,
            "projected_total_max_pf": round(completed + remaining, 2),
            "week_by_week": weeks_detail[tid],
        })

    results.sort(key=lambda r: r["projected_total_max_pf"])
    for i, r in enumerate(results, start=1):
        r["projected_inverse_draft_pick"] = i

    return {
        "league_id": league_id,
        "year": year,
        "league_name": base.get("settings", {}).get("name", ""),
        "current_week": current_week,
        "final_week": final_week,
        "final_week_auto_detected": final_week_override is None,
        "results": results,
        "sidebar": build_sidebar(base),
    }


def _roster_entries(team, week, source_id):
    """Build compute_max_pf's `entries` list for one team/week/stat-source."""
    entries = []
    for re in team.get("roster", {}).get("entries", []):
        ppe = re.get("playerPoolEntry", {})
        player = ppe.get("player", {})
        pts = player_week_points(player, week, source_id=source_id)
        eligible = set(player.get("eligibleSlots", []))
        name = player.get("fullName", "Unknown Player")
        entries.append({"name": name, "points": pts, "eligible_slots": eligible})
    return entries


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/compute")
def api_compute():
    league_id = request.args.get("leagueId", "").strip()
    year = request.args.get("year", "").strip()
    week = request.args.get("week", "").strip()
    espn_s2 = request.args.get("espn_s2", "").strip() or None
    swid = request.args.get("swid", "").strip() or None

    if not league_id or not year or not week:
        return jsonify({"error": "leagueId, year, and week are all required."}), 400

    try:
        year = int(year)
        week = int(week)
    except ValueError:
        return jsonify({"error": "year and week must be integers."}), 400

    try:
        result = compute_league(league_id, year, week, espn_s2, swid)
        return jsonify(result)
    except (PermissionError, LookupError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except requests.RequestException as e:
        return jsonify({"error": f"Network error reaching ESPN: {e}"}), 502


@app.route("/api/project")
def api_project():
    league_id = request.args.get("leagueId", "").strip()
    year = request.args.get("year", "").strip()
    final_week_raw = request.args.get("finalWeek", "").strip()
    espn_s2 = request.args.get("espn_s2", "").strip() or None
    swid = request.args.get("swid", "").strip() or None

    if not league_id or not year:
        return jsonify({"error": "leagueId and year are required."}), 400

    try:
        year = int(year)
        final_week_override = int(final_week_raw) if final_week_raw else None
    except ValueError:
        return jsonify({"error": "year and finalWeek must be integers."}), 400

    try:
        result = compute_season_projection(league_id, year, espn_s2, swid, final_week_override)
        return jsonify(result)
    except (PermissionError, LookupError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except requests.RequestException as e:
        return jsonify({"error": f"Network error reaching ESPN: {e}"}), 502


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ESPN Max PF / Inverse Draft Order</title>
<style>
  :root {
    --bg: #0f1216;
    --panel: #171b21;
    --border: #262c35;
    --text: #e8eaed;
    --muted: #8b93a1;
    --accent: #ff6a3d;
    --accent2: #2dd4bf;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px 16px 64px;
  }
  .wrap { max-width: 1180px; margin: 0 auto; }
  .layout { display: grid; grid-template-columns: 1fr; gap: 20px; }
  @media (min-width: 860px) {
    .layout { grid-template-columns: minmax(0, 1fr) 280px; align-items: start; }
  }
  .sidebar { display: flex; flex-direction: column; gap: 16px; }
  .sidebar .card { margin-bottom: 0; }
  .sidebar h2 { font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em;
                color: var(--muted); margin: 0 0 12px; }
  .champion-name { font-size: 1.15rem; font-weight: 700; color: var(--accent); }
  .champion-sub { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }
  .mtable { width: 100%; border-collapse: collapse; }
  .mtable td { padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 0.85rem; }
  .mtable tr:last-child td { border-bottom: none; }
  .mtable td.mlabel { color: var(--muted); font-size: 0.72rem; text-transform: uppercase;
                       letter-spacing: .03em; width: 44%; }
  .mtable td.mname { font-weight: 600; text-align: right; }
  .others-block { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); }
  .others-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .03em;
                   color: var(--muted); margin-bottom: 8px; }
  .others-list { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { background: #1d222b; border: 1px solid var(--border); border-radius: 20px;
          padding: 4px 10px; font-size: 0.78rem; }
  .placeholder { color: var(--muted); font-size: 0.8rem; font-style: italic; }
  h1 { font-size: 1.4rem; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 0.9rem; margin-bottom: 24px; }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
  }
  .row { display: flex; gap: 12px; flex-wrap: wrap; }
  .field { flex: 1; min-width: 110px; display: flex; flex-direction: column; gap: 6px; }
  label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  input {
    background: #0c0e12;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    color: var(--text);
    font-size: 0.95rem;
  }
  input:focus { outline: none; border-color: var(--accent); }
  details { margin-top: 14px; }
  summary { cursor: pointer; color: var(--muted); font-size: 0.85rem; }
  button {
    margin-top: 16px;
    background: var(--accent);
    color: #1a1a1a;
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    font-weight: 700;
    font-size: 0.95rem;
    cursor: pointer;
    width: 100%;
  }
  button:disabled { opacity: 0.5; cursor: default; }
  #status { margin-top: 12px; font-size: 0.88rem; color: var(--muted); min-height: 1.2em; }
  #error { color: #ff6b6b; font-size: 0.88rem; margin-top: 8px; display: none; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
  th { color: var(--muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: .03em; }
  tr.team-row { cursor: pointer; }
  tr.team-row:hover { background: #1d222b; }
  .pick { display: inline-block; background: var(--accent); color: #1a1a1a; font-weight: 700;
          border-radius: 6px; min-width: 24px; text-align: center; padding: 2px 6px; font-size: 0.8rem; }
  .maxpf { color: var(--accent2); font-weight: 700; }
  .left { color: var(--muted); }
  .lineup-panel { display: none; background: #12151b; border-top: 1px solid var(--border); }
  .lineup-panel.open { display: table-row; }
  .lineup-inner { padding: 12px 20px; }
  .lineup-inner table { margin-top: 0; }
  .lineup-inner th, .lineup-inner td { padding: 6px 8px; font-size: 0.83rem; }
  .hint { color: var(--muted); font-size: 0.8rem; margin-top: -8px; margin-bottom: 16px; }
  .tabs { display: flex; gap: 8px; margin-bottom: 18px; }
  .tab {
    flex: 1; text-align: center; padding: 10px; border-radius: 8px;
    border: 1px solid var(--border); background: #12151b; color: var(--muted);
    cursor: pointer; font-size: 0.85rem; font-weight: 600;
  }
  .tab.active { background: var(--accent); color: #1a1a1a; border-color: var(--accent); }
  .mode-panel { display: none; }
  .mode-panel.active { display: block; }
  .note { font-size: 0.8rem; color: var(--muted); margin-top: 10px; line-height: 1.4; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Max PF &amp; Inverse Draft Order</h1>
  <div class="sub">Enter a public ESPN fantasy football league and see each team's optimal-lineup ceiling — bench and IR players included.</div>

<div class="layout">
<div class="main">

  <div class="tabs">
    <div class="tab active" id="tab-week" onclick="setMode('week')">Single Week</div>
    <div class="tab" id="tab-season" onclick="setMode('season')">Season Projection</div>
  </div>

  <div class="card">
    <div class="mode-panel active" id="panel-week">
      <div class="row">
        <div class="field">
          <label for="leagueId">League ID</label>
          <input id="leagueId" placeholder="e.g. 123456" inputmode="numeric">
        </div>
        <div class="field">
          <label for="year">Season</label>
          <input id="year" placeholder="2025" inputmode="numeric" value="2025">
        </div>
        <div class="field">
          <label for="week">Week</label>
          <input id="week" placeholder="1" inputmode="numeric" value="1">
        </div>
      </div>
    </div>

    <div class="mode-panel" id="panel-season">
      <div class="row">
        <div class="field">
          <label for="leagueIdS">League ID</label>
          <input id="leagueIdS" placeholder="e.g. 123456" inputmode="numeric">
        </div>
        <div class="field">
          <label for="yearS">Season</label>
          <input id="yearS" placeholder="2026" inputmode="numeric" value="2026">
        </div>
        <div class="field">
          <label for="finalWeek">Final week (optional)</label>
          <input id="finalWeek" placeholder="auto-detect" inputmode="numeric">
        </div>
      </div>
      <div class="note">
        Sums each team's actual best-possible lineup for every week already played, then adds
        ESPN's own weekly projections against your <em>current</em> roster for every week left in the
        regular season. "Final week" auto-detects from your league's settings — override it if the
        guess looks wrong (playoff-only leagues can be tricky to detect).
      </div>
    </div>

    <details>
      <summary>Private league? (optional espn_s2 / SWID cookies)</summary>
      <div class="row" style="margin-top:10px;">
        <div class="field">
          <label for="espn_s2">espn_s2</label>
          <input id="espn_s2" placeholder="optional">
        </div>
        <div class="field">
          <label for="swid">SWID</label>
          <input id="swid" placeholder="optional, e.g. {ABC-123}">
        </div>
      </div>
    </details>
    <button id="runBtn" onclick="run()">Run</button>
    <div id="status"></div>
    <div id="error"></div>
  </div>

  <div id="resultsCard" class="card" style="display:none;">
    <h1 id="leagueName" style="font-size:1.1rem;"></h1>
    <div class="hint" id="resultsHint"></div>
    <table>
      <thead id="resultsHead"></thead>
      <tbody id="resultsBody"></tbody>
    </table>
  </div>

</div>

<aside class="sidebar">
  <div class="card">
    <h2>Current Champion</h2>
    <div class="champion-name">Steven Kotansky</div>
    <div class="champion-sub">Reigning league champ</div>
  </div>

  <div class="card">
    <h2>Marriage Tracker</h2>
    <table class="mtable">
      <tr><td class="mlabel">First Married</td><td class="mname">Justin Shaw</td></tr>
      <tr><td class="mlabel">Second Married</td><td class="mname">Steven Kotansky</td></tr>
      <tr><td class="mlabel">Engaged</td><td class="mname">Jackson Selby</td></tr>
    </table>
    <div class="others-block">
      <div class="others-label">Other</div>
      <div class="others-list" id="othersList">
        <span class="placeholder">Run a lookup to load the rest of the league's members.</span>
      </div>
    </div>
  </div>
</aside>
</div>

</div>

<script>
let mode = 'week';

function setMode(m) {
  mode = m;
  document.getElementById('tab-week').classList.toggle('active', m === 'week');
  document.getElementById('tab-season').classList.toggle('active', m === 'season');
  document.getElementById('panel-week').classList.toggle('active', m === 'week');
  document.getElementById('panel-season').classList.toggle('active', m === 'season');
  document.getElementById('resultsCard').style.display = 'none';
  document.getElementById('error').style.display = 'none';
  document.getElementById('status').textContent = '';
}

async function run() {
  if (mode === 'week') { await runWeek(); } else { await runSeason(); }
}

async function runWeek() {
  const leagueId = document.getElementById('leagueId').value.trim();
  const year = document.getElementById('year').value.trim();
  const week = document.getElementById('week').value.trim();
  const espn_s2 = document.getElementById('espn_s2').value.trim();
  const swid = document.getElementById('swid').value.trim();
  const btn = document.getElementById('runBtn');
  const status = document.getElementById('status');
  const errorEl = document.getElementById('error');
  const resultsCard = document.getElementById('resultsCard');

  errorEl.style.display = 'none';
  resultsCard.style.display = 'none';

  if (!leagueId || !year || !week) {
    errorEl.textContent = 'League ID, season, and week are all required.';
    errorEl.style.display = 'block';
    return;
  }

  btn.disabled = true;
  status.textContent = 'Fetching rosters and crunching optimal lineups...';

  const params = new URLSearchParams({ leagueId, year, week });
  if (espn_s2) params.set('espn_s2', espn_s2);
  if (swid) params.set('swid', swid);

  try {
    const resp = await fetch('/api/compute?' + params.toString());
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || 'Something went wrong.');
    }
    renderResults(data);
    status.textContent = 'Done.';
  } catch (e) {
    errorEl.textContent = e.message;
    errorEl.style.display = 'block';
    status.textContent = '';
  } finally {
    btn.disabled = false;
  }
}

async function runSeason() {
  const leagueId = document.getElementById('leagueIdS').value.trim();
  const year = document.getElementById('yearS').value.trim();
  const finalWeek = document.getElementById('finalWeek').value.trim();
  const espn_s2 = document.getElementById('espn_s2').value.trim();
  const swid = document.getElementById('swid').value.trim();
  const btn = document.getElementById('runBtn');
  const status = document.getElementById('status');
  const errorEl = document.getElementById('error');
  const resultsCard = document.getElementById('resultsCard');

  errorEl.style.display = 'none';
  resultsCard.style.display = 'none';

  if (!leagueId || !year) {
    errorEl.textContent = 'League ID and season are required.';
    errorEl.style.display = 'block';
    return;
  }

  btn.disabled = true;
  status.textContent = 'Fetching completed weeks + projections and crunching optimal lineups (this takes a bit longer)...';

  const params = new URLSearchParams({ leagueId, year });
  if (finalWeek) params.set('finalWeek', finalWeek);
  if (espn_s2) params.set('espn_s2', espn_s2);
  if (swid) params.set('swid', swid);

  try {
    const resp = await fetch('/api/project?' + params.toString());
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || 'Something went wrong.');
    }
    renderSeasonResults(data);
    status.textContent = 'Done.';
  } catch (e) {
    errorEl.textContent = e.message;
    errorEl.style.display = 'block';
    status.textContent = '';
  } finally {
    btn.disabled = false;
  }
}

function renderResults(data) {
  document.getElementById('leagueName').textContent =
    (data.league_name || ('League ' + data.league_id)) + ' — Week ' + data.week + ', ' + data.year;
  document.getElementById('resultsHint').textContent =
    "Tap a team row to see its optimal lineup for that week. Sorted by Max PF, low to high — that's your inverse draft order.";
  document.getElementById('resultsHead').innerHTML = `
    <tr><th>Pick</th><th>Team</th><th>Actual PF</th><th>Max PF</th><th>Left on bench</th></tr>
  `;

  const body = document.getElementById('resultsBody');
  body.innerHTML = '';

  data.results.forEach((r, idx) => {
    const rowId = 'lineup-' + idx;
    const tr = document.createElement('tr');
    tr.className = 'team-row';
    tr.onclick = () => toggleLineup(rowId);
    tr.innerHTML = `
      <td><span class="pick">${r.inverse_draft_pick}</span></td>
      <td>${r.team}</td>
      <td>${r.actual_pf}</td>
      <td class="maxpf">${r.max_pf}</td>
      <td class="left">+${r.points_left_on_bench}</td>
    `;
    body.appendChild(tr);

    const panelTr = document.createElement('tr');
    panelTr.className = 'lineup-panel';
    panelTr.id = rowId;
    const lineupRows = r.optimal_lineup.map(p =>
      `<tr><td>${p.slot}</td><td>${p.name}</td><td>${p.points}</td></tr>`
    ).join('');
    panelTr.innerHTML = `
      <td colspan="5">
        <div class="lineup-inner">
          <table>
            <thead><tr><th>Slot</th><th>Player</th><th>Points</th></tr></thead>
            <tbody>${lineupRows}</tbody>
          </table>
        </div>
      </td>
    `;
    body.appendChild(panelTr);
  });

  document.getElementById('resultsCard').style.display = 'block';
  updateSidebar(data.sidebar);
}

function renderSeasonResults(data) {
  document.getElementById('leagueName').textContent =
    (data.league_name || ('League ' + data.league_id)) + ' — ' + data.year + ' projected final standings';
  document.getElementById('resultsHint').textContent =
    `Currently week ${data.current_week}, regular season through week ${data.final_week}` +
    (data.final_week_auto_detected ? ' (auto-detected)' : ' (your override)') + '. ' +
    `Tap a team row for its week-by-week breakdown. Sorted by projected total Max PF, low to high.`;
  document.getElementById('resultsHead').innerHTML = `
    <tr><th>Proj. Pick</th><th>Team</th><th>Max PF to date</th><th>Proj. remaining</th><th>Proj. total</th></tr>
  `;

  const body = document.getElementById('resultsBody');
  body.innerHTML = '';

  data.results.forEach((r, idx) => {
    const rowId = 'lineup-' + idx;
    const tr = document.createElement('tr');
    tr.className = 'team-row';
    tr.onclick = () => toggleLineup(rowId);
    tr.innerHTML = `
      <td><span class="pick">${r.projected_inverse_draft_pick}</span></td>
      <td>${r.team}</td>
      <td>${r.cumulative_max_pf_to_date}</td>
      <td>${r.projected_remaining_max_pf}</td>
      <td class="maxpf">${r.projected_total_max_pf}</td>
    `;
    body.appendChild(tr);

    const panelTr = document.createElement('tr');
    panelTr.className = 'lineup-panel';
    panelTr.id = rowId;
    const weekRows = r.week_by_week.map(w =>
      `<tr><td>Wk ${w.week}</td><td>${w.type === 'actual' ? 'Actual' : 'Projected'}</td><td>${w.max_pf}</td></tr>`
    ).join('');
    panelTr.innerHTML = `
      <td colspan="5">
        <div class="lineup-inner">
          <table>
            <thead><tr><th>Week</th><th>Type</th><th>Max PF</th></tr></thead>
            <tbody>${weekRows}</tbody>
          </table>
        </div>
      </td>
    `;
    body.appendChild(panelTr);
  });

  document.getElementById('resultsCard').style.display = 'block';
  updateSidebar(data.sidebar);
}

function updateSidebar(sidebar) {
  if (!sidebar) return;
  const list = document.getElementById('othersList');
  const others = sidebar.marriage_tracker && sidebar.marriage_tracker.other || [];
  if (others.length === 0) {
    list.innerHTML = '<span class="placeholder">No other league members found.</span>';
    return;
  }
  list.innerHTML = others.map(name => `<span class="chip">${name}</span>`).join('');
}

function toggleLineup(id) {
  document.getElementById(id).classList.toggle('open');
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)