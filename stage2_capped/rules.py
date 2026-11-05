"""Decision rules for the Stage 2 movement-cap study (spec v4 section 8).

Pure functions over result rows (stage2_capped.row_schema). No model, no I/O.
Every cap-level and sweep-level conclusion is a label record carrying exactly
one of the four spec labels, with the fields the spec prints beside it.

Reading notes (each one is a reading of spec text; the scientific reviewer signs
off this file against section 8, see M2_RECEIPT.json "readings"):

- An *item* is a (start, seed pair): 10 starts x 2 seed pairs = 20 per cap. Seed
  pair k joins direct seed k with consistency seed k.
- An *arm result* is per checkpoint (arm and seed). A checkpoint is interpretable
  at a start only if it recovers (matched_reference) from the reference start at
  that start's own minimum, at that cap.
- A cap "covers" a connection or minimum if at least one interpretable item has it.
- A cap where neither arm recovers any interpretable original-start item (0 = 0)
  carries "uninformative (floor or gate)" (review S8, author's decision 2 Nov): there
  is no contrast to resolve. Priority: gate, floor, zero contrast, placebo.
- A cap flagged by the placebo check carries "uninformative (floor or gate)" with
  the note "not distinguishable from seed noise" (scientific review F1: the flag
  suppresses the cap's own apparent direction, so it must not read as an informative
  null). It is listed in not_evaluable and never contributes to a direction.
- The placebo item set is arm-level: starts where both seeds of the contrasted arm
  recover from the reference start at that start's own minimum (review F6).
- A table must hold every (cap incl. control) x checkpoint x start cell, or no label
  is produced (review F3). A missing lambda = 1 block does not block the primary
  labels; lambda = 1 is then reported "incomplete, not labelled" (review F4).
- Wording: "descriptive difference". The words "effect" and "significant" never
  appear in a record about the consistency-versus-direct comparison. The spec's own
  R-attribution phrase is quoted verbatim (it contains "effects").
"""
from stage2_capped.row_schema import (CAP_LABELS, CAP_VALUES, CONTROL_LABEL,
                                      EXPLORATORY_LABEL, LABELS,
                                      make_label_record, row_problems)

DIRECTION_STATED, SINGLE_CAP, NO_DIRECTION, UNINFORMATIVE = LABELS

CAP_ORDER = ("c0p05", "c0p10", "c0p25", "c1p00")
PAIRS = (0, 1)
PRIMARY_LAMBDA = 0.1
EXPLORATORY_LAMBDA = 1.0
N_STARTS = 10
N_CONNECTIONS = 5
N_REFERENCE_STARTS = 6
CONTROL_RUNS = 80
MIN_CONNECTIONS = 3
MIN_MINIMA = 2
MIN_PAIR_CONNECTIONS = 2
JUMP_THRESHOLD_A = 1.0
MIN_ATTRIBUTION_RUNS = 2

CAVEATS = ("the unit of analysis is five connections, not ten independent starts",
           "the sweep is a robustness check on cap choice, not replication: the caps "
           "reuse the same starts, and 0.05 and 0.10 bind on the same first trials",
           "a descriptive difference only")

NOT_ATTRIBUTABLE = "not attributable to jump removal; consistent with step-size effects"
ATTRIBUTION_STATUSES = ("supported", "not excluded", NOT_ATTRIBUTABLE,
                        "no new recovery to attribute")

# Each cap-level record has one of these statuses (detail["status"]).
STATUS_GATE, STATUS_FLOOR, STATUS_ZERO, STATUS_PLACEBO, STATUS_ELIGIBLE = (
    "gate", "floor", "zero_contrast", "placebo", "eligible")


class TableError(ValueError):
    """The result table is incomplete or internally inconsistent. Nothing is
    labelled from a table that cannot be trusted."""


# ---- Checkpoints --------------------------------------------------------

def checkpoints(consistency_lambda):
    """The checkpoint identities (arm, seed, lambda) one comparison uses."""
    return {"energy_s0": ("energy", 0, None),
            "direct_s0": ("direct", 0, None), "direct_s1": ("direct", 1, None),
            "consistency_s0": ("consistency", 0, consistency_lambda),
            "consistency_s1": ("consistency", 1, consistency_lambda)}


def _ckpt(row):
    return (row["arm"], row["seed"], row["consistency_lambda"])


def _direct(pair):
    return ("direct", pair, None)


def _consistency(pair, lam):
    return ("consistency", pair, lam)


# ---- The result table ---------------------------------------------------

class Table:
    """Rows indexed by (cap label, checkpoint, start kind, start id)."""

    def __init__(self, rows):
        self.by_key = {}
        for row in rows:
            problems = row_problems(row)
            if problems:
                raise TableError("row breaks the schema: " + "; ".join(problems))
            key = (row["cap_label"], _ckpt(row), row["start_kind"], row["start_id"])
            if key in self.by_key:
                raise TableError(f"duplicate row for {key}")
            self.by_key[key] = row
        self._collect_starts()
        self._check_unbound_equals_control()

    def _collect_starts(self):
        originals, references = {}, {}
        for (_, _, kind, start_id), row in self.by_key.items():
            if kind == "original":
                if row["connection"] is None:
                    raise TableError(f"original start {start_id} has no connection")
                info = (row["connection"], row["branch"], row["own_minimum"])
                if originals.setdefault(start_id, info) != info:
                    raise TableError(f"start {start_id} differs between rows")
            else:
                if references.setdefault(start_id, row["own_minimum"]) != row["own_minimum"]:
                    raise TableError(f"reference start {start_id} differs between rows")
        self.starts = [{"start_id": s, "connection": i[0], "branch": i[1], "minimum": i[2]}
                       for s, i in sorted(originals.items())]
        self.reference_by_minimum = {}
        for start_id, minimum in sorted(references.items()):
            if minimum in self.reference_by_minimum:
                raise TableError(f"two reference starts for minimum {minimum}")
            self.reference_by_minimum[minimum] = start_id

    def _check_unbound_equals_control(self):
        """A capped run whose cap never bound equals its control by construction.
        A difference means the harness is broken, so no label is produced."""
        for (cap, ckpt, kind, start_id), row in self.by_key.items():
            if cap == CONTROL_LABEL or row["capped_trial_count"] != 0:
                continue
            control = self.by_key.get((CONTROL_LABEL, ckpt, kind, start_id))
            if control is None:
                continue
            for field in ("outcome_category", "stop_reason", "step_count",
                          "force_evaluation_count"):
                if control[field] != row[field]:
                    raise TableError(
                        f"{cap} run {ckpt} {start_id} never bound but differs from its "
                        f"control in {field}")

    def row(self, cap, ckpt, kind, start_id):
        try:
            return self.by_key[(cap, ckpt, kind, start_id)]
        except KeyError:
            raise TableError(f"missing row: cap {cap}, checkpoint {ckpt}, "
                             f"{kind} start {start_id}") from None

    def has(self, cap, ckpt, kind, start_id):
        return (cap, ckpt, kind, start_id) in self.by_key

    def original_recovered(self, cap, ckpt, start_id):
        return self.row(cap, ckpt, "original", start_id)["recovered"]

    def reference_recovered(self, cap, ckpt, minimum):
        if minimum not in self.reference_by_minimum:
            raise TableError(f"no reference start for minimum {minimum}")
        return self.row(cap, ckpt, "reference",
                        self.reference_by_minimum[minimum])["recovered"]

    @property
    def connections(self):
        return sorted({s["connection"] for s in self.starts})

    def check_shape(self):
        if len(self.starts) != N_STARTS:
            raise TableError(f"expected {N_STARTS} original starts, found {len(self.starts)}")
        if len(self.connections) != N_CONNECTIONS:
            raise TableError(f"expected {N_CONNECTIONS} connections, "
                             f"found {len(self.connections)}")
        if len(self.reference_by_minimum) != N_REFERENCE_STARTS:
            raise TableError(f"expected {N_REFERENCE_STARTS} reference starts, "
                             f"found {len(self.reference_by_minimum)}")
        for start in self.starts:
            if start["minimum"] not in self.reference_by_minimum:
                raise TableError(f"start {start['start_id']} has own minimum "
                                 f"{start['minimum']} with no reference start")

    def check_complete(self, checkpoint_keys, caps=CAP_ORDER):
        """Every (cap incl. control) x checkpoint x start cell must exist. A missing
        cell silently lowers a count, so a table with one gets no labels."""
        self.check_shape()
        original_ids = [s["start_id"] for s in self.starts]
        reference_ids = [self.reference_by_minimum[m] for m in sorted(self.reference_by_minimum)]
        missing = [(cap, ckpt, kind, start_id)
                   for cap in (CONTROL_LABEL,) + tuple(caps)
                   for ckpt in checkpoint_keys
                   for kind, ids in (("original", original_ids), ("reference", reference_ids))
                   for start_id in ids
                   if (cap, ckpt, kind, start_id) not in self.by_key]
        if missing:
            raise TableError(f"incomplete table: {len(missing)} missing cells, "
                             f"first {missing[0]}")


def lambda_one_checkpoints():
    return [("consistency", 0, EXPLORATORY_LAMBDA), ("consistency", 1, EXPLORATORY_LAMBDA)]


def has_lambda_one(table):
    return any(ckpt[2] == EXPLORATORY_LAMBDA for (_, ckpt, _, _) in table.by_key)


# ---- R-control ----------------------------------------------------------

def r_control(compared, mismatches, frozen_rerun=None, expected_runs=CONTROL_RUNS):
    """Stage 0 must reproduce the saved results exactly.

    compared: runs compared so far. mismatches: ordered list of task ids that
    differ from the saved result (fail fast, so at most one). frozen_rerun: None
    or {"task": id, "matches_saved": bool}, the one pre-authorized re-run of that
    task with the frozen relax in the same runtime.
    proceed is True only for an exact, complete match. No tolerance exists."""
    def verdict(name, proceed=False, action=None):
        return {"verdict": name, "proceed": proceed, "action": action}

    if not mismatches:
        if compared == expected_runs:
            return verdict("pass", proceed=True)
        return verdict("stop: incomplete comparison",
                       action=f"compared {compared} of {expected_runs} runs")
    if len(mismatches) > 1:
        return verdict("stop: unauthorized continuation",
                       action="fail fast was violated; a new reviewed launch record is needed")
    if frozen_rerun is None:
        return verdict("frozen re-run required",
                       action=f"run the frozen relax once on {mismatches[0]}")
    if frozen_rerun.get("task") != mismatches[0]:
        return verdict("stop: unauthorized continuation",
                       action="the pre-authorized re-run covers only the first mismatching task")
    if not frozen_rerun.get("matches_saved"):
        return verdict("stop: runtime drift",
                       action="the frozen relax also differs from the saved result")
    return verdict("stop: code error",
                   action="the frozen relax matches the saved result but c = infinity does not")


# ---- R-adequacy, items, R-support, R-direction, placebo -----------------

def adequacy(table, cap):
    """Energy C seed 0 must recover at least 1 of the 10 original starts at this
    cap, and more than at the control (c = infinity)."""
    energy = checkpoints(PRIMARY_LAMBDA)["energy_s0"]
    at_cap = sum(table.original_recovered(cap, energy, s["start_id"]) for s in table.starts)
    at_control = sum(table.original_recovered(CONTROL_LABEL, energy, s["start_id"])
                     for s in table.starts)
    return {"recovered_at_cap": at_cap, "recovered_at_control": at_control,
            "starts": len(table.starts), "passed": at_cap >= 1 and at_cap > at_control}


def items_at_cap(table, cap, lam):
    """Interpretable items at a cap (R-floor), and the floor items left out."""
    items, floor = [], []
    for start in table.starts:
        for pair in PAIRS:
            direct, consistency = _direct(pair), _consistency(pair, lam)
            direct_ok = table.reference_recovered(cap, direct, start["minimum"])
            consistency_ok = table.reference_recovered(cap, consistency, start["minimum"])
            if direct_ok and consistency_ok:
                items.append({
                    "start_id": start["start_id"], "connection": start["connection"],
                    "minimum": start["minimum"], "pair": pair,
                    "direct": int(table.original_recovered(cap, direct, start["start_id"])),
                    "consistency": int(table.original_recovered(cap, consistency,
                                                                start["start_id"]))})
            else:
                floor.append({"start_id": start["start_id"], "pair": pair,
                              "direct_interpretable": direct_ok,
                              "consistency_interpretable": consistency_ok})
    return items, floor


def coverage(items):
    """Connections and minima covered, and connections covered per seed pair."""
    return {"items": len(items),
            "connections": len({i["connection"] for i in items}),
            "minima": len({i["minimum"] for i in items}),
            "pair_connections": {f"pair_{p}": len({i["connection"] for i in items
                                                   if i["pair"] == p}) for p in PAIRS}}


def support(items):
    """R-support: at least 3 connections and 2 minima, and each seed pair covers
    at least 2 connections."""
    cov = coverage(items)
    reasons = []
    if cov["connections"] < MIN_CONNECTIONS:
        reasons.append(f"fewer than {MIN_CONNECTIONS} connections covered")
    if cov["minima"] < MIN_MINIMA:
        reasons.append(f"fewer than {MIN_MINIMA} own minima covered")
    for name, count in cov["pair_connections"].items():
        if count < MIN_PAIR_CONNECTIONS:
            reasons.append(f"{name} covers fewer than {MIN_PAIR_CONNECTIONS} connections")
    return {"ok": not reasons, "reasons": reasons, **cov}


def _sign(value):
    return (value > 0) - (value < 0)


def direction(items, connections):
    """R-direction on a set of interpretable items. Two-sided: it may favor direct.

    Per connection, sum (consistency - direct) recoveries over its items. Direction
    holds only if at least 3 of the connections favor one arm, none favor the
    other, and both seed pairs have a nonzero total difference of that sign."""
    sums = {c: 0 for c in connections}
    pair_totals = {p: 0 for p in PAIRS}
    for item in items:
        diff = item["consistency"] - item["direct"]
        sums[item["connection"]] += diff
        pair_totals[item["pair"]] += diff
    n_consistency = sum(1 for v in sums.values() if v > 0)
    n_direct = sum(1 for v in sums.values() if v < 0)
    favors = None
    if n_consistency >= MIN_CONNECTIONS and n_direct == 0:
        favors = "consistency"
    elif n_direct >= MIN_CONNECTIONS and n_consistency == 0:
        favors = "direct"
    pairs_agree = False
    if favors is not None:
        sign = 1 if favors == "consistency" else -1
        pairs_agree = all(_sign(t) == sign for t in pair_totals.values())
    recoveries_direct = sum(i["direct"] for i in items)
    recoveries_consistency = sum(i["consistency"] for i in items)
    return {"holds": favors is not None and pairs_agree,
            "favors": favors if pairs_agree else None,
            "recoveries_direct": recoveries_direct,
            "recoveries_consistency": recoveries_consistency,
            "zero_contrast": recoveries_direct == 0 and recoveries_consistency == 0,
            "connection_sums": dict(sums),
            "connections_favoring_consistency": n_consistency,
            "connections_favoring_direct": n_direct,
            "pair_totals": {f"pair_{p}": t for p, t in pair_totals.items()},
            "items": len(items)}


def placebo(table, cap, lam):
    """Seed-noise check: the connection part of R-direction applied to the
    within-arm contrasts (direct seed 0 vs seed 1, consistency seed 0 vs seed 1),
    on starts where both seeds of that arm are interpretable at this cap."""
    result = {"flagged": False, "contrasts": {}}
    for arm, make in (("direct", _direct),
                      ("consistency", lambda pair: _consistency(pair, lam))):
        seed0, seed1 = make(0), make(1)
        sums = {c: 0 for c in table.connections}
        used = 0
        for start in table.starts:
            if (table.reference_recovered(cap, seed0, start["minimum"]) and
                    table.reference_recovered(cap, seed1, start["minimum"])):
                used += 1
                sums[start["connection"]] += (
                    int(table.original_recovered(cap, seed1, start["start_id"])) -
                    int(table.original_recovered(cap, seed0, start["start_id"])))
        n_seed1 = sum(1 for v in sums.values() if v > 0)
        n_seed0 = sum(1 for v in sums.values() if v < 0)
        flagged = ((n_seed1 >= MIN_CONNECTIONS and n_seed0 == 0) or
                   (n_seed0 >= MIN_CONNECTIONS and n_seed1 == 0))
        result["contrasts"][arm] = {
            "starts_used": used, "connection_sums": dict(sums),
            "connections_favoring_seed_1": n_seed1,
            "connections_favoring_seed_0": n_seed0, "flagged": flagged}
        result["flagged"] = result["flagged"] or flagged
    return result


def reference_counts(table, cap, lam):
    counts = {name: sum(table.reference_recovered(cap, ckpt, m)
                        for m in sorted(table.reference_by_minimum))
              for name, ckpt in checkpoints(lam).items()}
    counts["reference_starts"] = len(table.reference_by_minimum)
    return counts


# ---- Cap assessment and the sweep ---------------------------------------

def assess_cap(table, cap, lam):
    """Everything R-adequacy, R-floor, R-support, R-direction and the placebo
    check say about one cap. status, in priority order: gate, floor, zero_contrast,
    placebo, or eligible."""
    gate = adequacy(table, cap)
    items, floor_items = items_at_cap(table, cap, lam)
    sup = support(items)
    direction_here = direction(items, table.connections)
    noise = placebo(table, cap, lam)
    if not gate["passed"]:
        status = STATUS_GATE
    elif not sup["ok"]:
        status = STATUS_FLOOR
    elif direction_here["zero_contrast"]:
        status = STATUS_ZERO
    elif noise["flagged"]:
        status = STATUS_PLACEBO
    else:
        status = STATUS_ELIGIBLE
    return {"cap": cap, "status": status, "adequacy": gate, "items": items,
            "floor_items": floor_items, "support": sup, "direction": direction_here,
            "placebo": noise, "reference_counts": reference_counts(table, cap, lam)}


def _keys(items):
    return {(i["start_id"], i["pair"]) for i in items}


def sweep(table, assessments, lam):
    """R-sweep over the caps' assessments. Returns the sweep label, its reason and
    the evidence behind it."""
    E = [a["cap"] for a in assessments if a["status"] == STATUS_ELIGIBLE]
    by_cap = {a["cap"]: a for a in assessments}
    detail = {"set_E": list(E), "favors": None}
    if not E:
        detail["r_failure"] = True
        detail["reason"] = ("no cap is evaluable: the procedure remained uninformative for "
                            "the comparison, so no consistency conclusion is drawn")
        return UNINFORMATIVE, detail, [], 0
    detail["r_failure"] = False
    if len(E) == 1:
        own = by_cap[E[0]]["direction"]
        detail["reason"] = ("one cap is evaluable: a single-cap observation, not a "
                            "robustness check on cap choice")
        detail["r_direction_holds_at_the_cap"] = own["holds"]
        detail["favors_at_the_cap"] = own["favors"]
        return SINGLE_CAP, detail, by_cap[E[0]]["items"], len(by_cap[E[0]]["items"])

    own = {cap: by_cap[cap]["direction"] for cap in E}
    detail["own_items"] = {cap: {"holds": d["holds"], "favors": d["favors"]}
                           for cap, d in own.items()}
    common_keys = set.intersection(*[_keys(by_cap[cap]["items"]) for cap in E])
    common = {cap: [i for i in by_cap[cap]["items"]
                    if (i["start_id"], i["pair"]) in common_keys] for cap in E}
    common_items = common[E[0]]
    if not all(d["holds"] for d in own.values()):
        detail["reason"] = "R-direction does not hold at every cap in E"
        return NO_DIRECTION, detail, common_items, len(common_items)
    arms = {d["favors"] for d in own.values()}
    if len(arms) != 1:
        detail["reason"] = "the descriptive difference has opposite signs across caps"
        return NO_DIRECTION, detail, common_items, len(common_items)
    arm = arms.pop()
    common_support = support(common_items)
    detail["common_items"] = {"items": len(common_items), "support_ok": common_support["ok"],
                              "support_reasons": common_support["reasons"]}
    if not common_support["ok"]:
        detail["reason"] = ("the items interpretable at every cap in E fail R-support, so "
                            "robustness to the item set cannot be assessed")
        return UNINFORMATIVE, detail, common_items, len(common_items)
    common_dir = {cap: direction(common[cap], table.connections) for cap in E}
    detail["common_items"]["holds"] = {cap: d["holds"] for cap, d in common_dir.items()}
    if all(d["holds"] and d["favors"] == arm for d in common_dir.values()):
        detail["reason"] = (f"R-direction holds at every cap in E on each cap's own items "
                            f"and on the common items; descriptive difference favors {arm}")
        detail["favors"] = arm
        return DIRECTION_STATED, detail, common_items, len(common_items)
    detail["reason"] = "varies with item set"
    return NO_DIRECTION, detail, common_items, len(common_items)


def _cap_detail(assessment):
    """The structured evidence printed beside a cap-level label."""
    return {"status": assessment["status"], "adequacy": assessment["adequacy"],
            "support": assessment["support"], "direction": assessment["direction"],
            "placebo": assessment["placebo"],
            "floor_items": assessment["floor_items"]}


def evaluate_comparison(table, lam=PRIMARY_LAMBDA, caps=CAP_ORDER):
    """Label every cap and the sweep for direct versus consistency (lambda = lam).

    lambda = 1 is reported the same way and carries the exploratory label."""
    table.check_complete(list(checkpoints(lam).values()), caps)
    assessments = [assess_cap(table, cap, lam) for cap in caps]
    sweep_label, sweep_detail, sweep_items, sweep_n = sweep(table, assessments, lam)
    E = sweep_detail["set_E"]
    not_evaluable = [a["cap"] for a in assessments
                     if a["adequacy"]["passed"] and a["cap"] not in E]
    exploratory = {"exploratory_label": EXPLORATORY_LABEL} if lam == EXPLORATORY_LAMBDA else {}

    cap_records = []
    for a in assessments:
        if a["status"] == STATUS_ELIGIBLE:
            label = sweep_label
            note = "conclusion is the sweep's (see the sweep record)"
        elif a["status"] == STATUS_PLACEBO:
            label = UNINFORMATIVE
            note = ("not distinguishable from seed noise; the cap leaves E and "
                    "supports no direction")
        elif a["status"] == STATUS_ZERO:
            label = UNINFORMATIVE
            note = (f"neither arm recovered any interpretable original-start item "
                    f"(0 of {a['support']['items']}); the comparison has no contrast "
                    f"to resolve, so this is a floor on the original starts, not a "
                    f"finding that the arms are alike")
        elif a["status"] == STATUS_FLOOR:
            label = UNINFORMATIVE
            note = "floor: " + "; ".join(a["support"]["reasons"])
        elif a["status"] == STATUS_GATE:
            label = UNINFORMATIVE
            note = "gate: the energy C seed 0 baseline does not pass R-adequacy"
        else:
            raise AssertionError(f"unknown cap status {a['status']!r}")
        cov = a["support"]
        detail = {**_cap_detail(a), "note": note, "comparison_lambda": lam,
                  "favors_own_items": a["direction"]["favors"], **exploratory}
        cap_records.append(make_label_record(
            scope="cap", cap_label=a["cap"], label=label,
            interpretable_items=cov["items"], connections_covered=cov["connections"],
            minima_covered=cov["minima"], adequacy_passed=a["adequacy"]["passed"],
            reference_start_counts=a["reference_counts"], set_E=list(E),
            not_evaluable=list(not_evaluable), detail=detail))

    sweep_cov = coverage(sweep_items)
    flat_counts = {f"{a['cap']}/{name}": n for a in assessments
                   for name, n in a["reference_counts"].items()}
    sweep_record = make_label_record(
        scope="sweep", cap_label=None, label=sweep_label,
        interpretable_items=sweep_cov["items"], connections_covered=sweep_cov["connections"],
        minima_covered=sweep_cov["minima"], adequacy_passed=None,
        reference_start_counts=flat_counts, set_E=list(E), not_evaluable=list(not_evaluable),
        detail={**sweep_detail, **exploratory, "comparison_lambda": lam,
                "caveats": list(CAVEATS),
                "zero_contrast_caps": [a["cap"] for a in assessments
                                       if a["direction"]["zero_contrast"]],
                "per_cap": {a["cap"]: {"status": a["status"],
                                       "interpretable_items": a["support"]["items"],
                                       "connections_covered": a["support"]["connections"],
                                       "minima_covered": a["support"]["minima"],
                                       "adequacy_passed": a["adequacy"]["passed"],
                                       "recoveries_direct":
                                           a["direction"]["recoveries_direct"],
                                       "recoveries_consistency":
                                           a["direction"]["recoveries_consistency"],
                                       "zero_contrast": a["direction"]["zero_contrast"],
                                       "favors_own_items": a["direction"]["favors"]}
                            for a in assessments}})
    return {"lambda": lam, "cap_records": cap_records, "sweep_record": sweep_record,
            "assessments": assessments}


# ---- R-attribution ------------------------------------------------------

def _is_primary(ckpt):
    arm, _, lam = ckpt
    return arm != "consistency" or lam == PRIMARY_LAMBDA


def r_attribution(table, include_lambda1=None):
    """Is jump removal supported as the reason capped runs recover?

    Supported only if at least two original-start runs at c = 1.0 recover although
    the same checkpoint and start did not at c = infinity, each such run's cap
    binds, and its c = infinity run has an accepted step above 1.0 angstrom. The
    condition is necessary, not sufficient.

    lambda = 1 runs count but are labelled exploratory: whenever one contributes to
    a count, the status text itself carries the label (base_status is the plain
    value). include_lambda1 None means "if the table has lambda = 1 rows"; when it
    is included its grid must be complete. Every checkpoint used needs its full
    (cap incl. control) x start grid, so a missing control row raises."""
    if include_lambda1 is None:
        include_lambda1 = has_lambda_one(table)
    keys = list(checkpoints(PRIMARY_LAMBDA).values())
    if include_lambda1:
        keys += lambda_one_checkpoints()
    table.check_complete(keys)
    used = set(keys)

    new_by_cap = {cap: [] for cap in CAP_ORDER}
    qualifying, non_qualifying = [], []
    for (cap, ckpt, kind, start_id), row in sorted(
            table.by_key.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][3])):
        if (kind != "original" or cap == CONTROL_LABEL or ckpt not in used
                or not row["recovered"]):
            continue
        control = table.row(CONTROL_LABEL, ckpt, "original", start_id)
        if control["recovered"]:
            continue
        flip = {"checkpoint_id": row["checkpoint_id"], "start_id": start_id,
                "primary": _is_primary(ckpt)}
        new_by_cap[cap].append(flip)
        if cap == "c1p00":
            binds = row["capped_trial_count"] > 0
            big_step = (control["max_accepted_displacement_A"] is not None and
                        control["max_accepted_displacement_A"] > JUMP_THRESHOLD_A)
            if binds and big_step:
                qualifying.append(flip)
            else:
                failed = ([] if binds else ["the cap never bound"]) + \
                    ([] if big_step else ["the c = infinity run has no accepted step above 1.0 A"])
                non_qualifying.append({**flip, "failed": failed})
    n_primary = sum(1 for q in qualifying if q["primary"])
    new_at_one = new_by_cap["c1p00"]
    new_small = [r for cap in ("c0p05", "c0p10", "c0p25") for r in new_by_cap[cap]]
    if len(qualifying) >= MIN_ATTRIBUTION_RUNS:
        base = "supported"
    elif new_at_one:
        base = "not excluded"
    elif new_small:
        base = NOT_ATTRIBUTABLE
    else:
        base = "no new recovery to attribute"
    lambda1_contributes = any(not r["primary"] for flips in new_by_cap.values() for r in flips)
    result = {"status": base, "base_status": base, "qualifying_runs": qualifying,
              "non_qualifying_flips": non_qualifying,
              "qualifying_primary": n_primary,
              "qualifying_lambda1": len(qualifying) - n_primary,
              "new_recoveries_by_cap": {cap: len(v) for cap, v in new_by_cap.items()},
              "lambda1_included": bool(include_lambda1),
              "necessary_not_sufficient": True}
    if lambda1_contributes:
        result["status"] = (f"{base} ({EXPLORATORY_LABEL}: lambda = 1 runs contribute; "
                            f"primary runs alone qualifying: {n_primary})")
        result["exploratory_label"] = EXPLORATORY_LABEL
        result["supported_by_primary_runs_alone"] = n_primary >= MIN_ATTRIBUTION_RUNS
    return result


# ---- Top level ----------------------------------------------------------

def analyze(rows):
    """All labels from saved rows: the primary comparison (lambda = 0.1), the
    exploratory comparison (lambda = 1) when its rows exist and are complete, and
    R-attribution. A primary table that is incomplete or inconsistent raises. An
    incomplete lambda = 1 block (a Stage 2 kill, say) does not cost the primary
    labels: it is reported "incomplete, not labelled" with the reason."""
    table = Table(rows)
    out = {"primary": evaluate_comparison(table, PRIMARY_LAMBDA)}
    lambda1_complete = False
    if has_lambda_one(table):
        try:
            out["exploratory"] = evaluate_comparison(table, EXPLORATORY_LAMBDA)
            lambda1_complete = True
        except TableError as error:
            out["exploratory"] = {"status": "incomplete, not labelled", "reason": str(error),
                                  "exploratory_label": EXPLORATORY_LABEL}
    out["attribution"] = r_attribution(table, include_lambda1=lambda1_complete)
    return out
