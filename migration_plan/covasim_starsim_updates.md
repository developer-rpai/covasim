# Starsim updates needed for the Covasim v4 port

Changes that belong in Starsim rather than as workarounds in Covasim (per Cliff, 2026-09-29: a quick Starsim fix is much better than a hack in Covasim). Companion to `port_review_claude.md`. References are to Starsim 3.6.1 (`/home/cliffk/idm/starsim/starsim`) and the Covasim `starsim-port` branch. Several of these were also needed by HPVsim, so they should help other disease libraries as well.

Priority: **P1** blocks v3 compatibility or causes silent failures; **P2** removes significant duplication in Covasim; **P3** nice to have.

## P1

### 1. Integer-like comparisons on `ss.Timeline`

- **Problem:** v3 custom interventions and analyzers use `sim.t` as the integer day (`if sim.t == 10`, `if sim.t < 50`, `sim.t % 7`, `results[k][sim.t]`). In v4 `sim.t` is an `ss.Timeline`, so `==` is always False (the intervention silently never fires) and `<` raises TypeError. This is the most common v3 customization pattern and the worst compat break found.
- **Proposal:** give `ss.Timeline` `__eq__`/`__ne__`/`__lt__`/`__le__`/`__gt__`/`__ge__` against plain ints (and numpy ints) that compare to `self.ti`, plus `__index__` and `__int__` returning `ti` (so `arr[sim.t]` and `range(sim.t)` work), and `__hash__` if needed. Comparison with `ss.date`/`ss.dur` could compare to `now`, and anything else raises a clear error.
- **Open question:** this makes `sim.t == 10` mean "timestep 10", which equals "day 10" only when `dt=1 day` (always true in Covasim). For Starsim sims with other `dt`, that may surprise users; alternatives are to only enable it when the unit is days, or to warn when `dt` is not 1. `Timeline.__len__` and `__bool__` already exist (`timeline.py:184-188`), so check `bool(sim.t)` semantics don't conflict.
- **Covasim side:** nothing, once this lands.

### 2. Allow duplicate module types with automatic unique names

- **Problem:** v3 allows two interventions of the same type, e.g. two `vaccinate_prob` for primary series plus booster, or two `vaccinate_num`. In v4 the second raises `Cannot add object "vaccinate_num" since already present in ndict` (`utils.py:89`).
- **Proposal:** when adding modules to a sim (interventions and analyzers at least), auto-suffix duplicate default names (`vaccinate_num`, `vaccinate_num_1`, …) instead of raising. Only raise if the user explicitly set the same `name` twice.

### 3. Reserved attribute names on user modules

- **Problem:** `ss.Module` locks `pars`, `t`, `sim`, `dists`, `results` (`modules.py:273`). v3 tutorial analyzers do `self.t = []` in `__init__` (e.g. the SEIR analyzer), which now raises. Separately, a user module named `a` clashes with Covasim's random network, also named `a` (network and analyzer names share a namespace).
- **Options:**
  - (a) Let module names be unique per module type (interventions, analyzers, networks) rather than globally.
  - (b) For `t`: nothing in Starsim, and it becomes a documented migration rule (`self.t` → e.g. `self.tvec`). Changing `t` itself would be a large Starsim change.
- **Covasim side:** migration rule plus a clear error message if (b).

### 4. Time-varying population scale (for dynamic rescaling)

- **Problem:** v3's default `rescale=True` starts with a small `pop_scale` and increases it as susceptibles get depleted, "making naive" a fraction of agents each time. Starsim scales `scale=True` results once, at finalize, by a single `pars.pop_scale` (`sim.py:568`, `modules.py:880`). Dynamic rescaling needs a per-timestep scale factor.
- **Proposal:** support a scale factor that can change over the run: e.g. `sim.results.pop_scale` as a per-timestep Result (defaulting to constant `pars.pop_scale`), and have result scaling multiply by it elementwise, for flows at each step and for stocks by the scale at that time. Covasim then implements the rescaling logic itself (deciding when to rescale and which agents become naive) as a small module that writes the scale factor.
- **Covasim side:** the rescale module plus `rescale`, `rescale_threshold`, `rescale_factor`, `pop_scale` pars.

### 5. Network edge manipulation for the Covasim layer API

- **Problem:** Cliff wants the v3 layer API ported: `people.contacts['h']`, `cv.Layer`, `add_layer`, `dynam_layer`, `layer.find_contacts`, removing/adding edges, and per-edge `beta`. Covasim can build most of this on `ss.Network.edges`, but a few generic operations are missing or awkward.
- **Proposal:** before writing the Covasim shim, check which of these `ss.Network` already has, and add any that are missing:
  - append edges (with optional per-edge beta)
  - remove edges by boolean mask or index
  - build a network from a v3-style contacts dict or dataframe (`p1`, `p2`, `beta`)
  - export edges as a dataframe
  - a `dynamic` flag for a network that is regenerated each step versus static
- **Covasim side:** `cv.Layer` as a thin view over `ss.Network`; `people.contacts` as a dict-like view over `sim.networks`.

## P2

### 6. Multi-dimensional results (or a standard "results by group" pattern)

- **Problem:** Covasim's 12 by-variant results are 2D (time × variant) in a nested `ss.Results`. Starsim's auto-scaling, `flatten`, `to_df`, `MultiSim.reduce` and `plot` don't handle them, which caused ~300 lines of parallel machinery in Covasim (manual `pop_scale` scaling in `sim.finalize`, a custom `sim.plot`, `to_excel`, MultiSim reduce) and crashes `sim.to_df()`/`to_json()`. HPVsim needed the same thing for genotypes, and by-age results are the same pattern.
- **Options:**
  - (a) First-class 2D `ss.Result` (a second labeled axis), supported in scaling, `flatten`/`to_df` (one column per group), `MultiSim.reduce`, and plotting (one line per group).
  - (b) Lighter: a helper that defines a group of 1D results from one spec (`define_results_by(..., groups=['wild','alpha'])`) plus a view that stacks them into 2D on demand.
  - Either lets Covasim delete its custom results stack.

### 7. Hook in `ss.Infection.infect` for per-strain transmission

- **Problem:** `cv.COVID.infect()` (`covid.py:516-618`) copies the whole network loop of `ss.Infection.infect` (`diseases.py:255`) to add an outer loop over variants with per-variant `rel_trans`/`rel_sus`/beta, then dedups (lowest variant index wins, matching v3). Any change upstream in `infect()` (e.g. the recent raw-array optimization) must be manually mirrored.
- **Proposal:** split `infect()` so the per-network part is a method, e.g. `infect_route(route, rel_trans, rel_sus, betamap)` returning `(targets, sources, network_ids)`, and `infect()` just computes `rel_trans`/`rel_sus` and loops routes. A multi-strain disease then overrides `infect()` with a small loop over strains calling `infect_route`. Alternatively, a documented `compute_rel_trans()`/`compute_rel_sus()` hook pair.

### 8. Growth-aware multi-column agent arrays

- **Problem:** Covasim's per-variant immunity (`sus_imm`, `symp_imm`, `sev_imm`: agents × variants) are plain ndarrays, so they don't grow with the population, which blocks births (`covid.py:1075-1083`). HPVsim had the same need for genotypes.
- **Proposal:** an `ss.FloatArr` with an extra fixed-size axis (`ss.FloatArr('sus_imm', shape=nv)`) that grows along the agent axis like other states. Fallback in Covasim: one `ss.FloatArr` per variant, which works but is clunkier.

### 9. Exact-count seeding

- **Problem:** Covasim's `pop_infected=20` means exactly 20 agents infected at t=0; `init_prev` is a Bernoulli probability (`diseases.py:152-155`). Covasim hand-rolls this with its own seeded numpy RNG (`covid.py:1107`). HPVsim has similar code in `seeding.py`.
- **Proposal:** let `init_prev` accept an int (or `ss.count(n)`) meaning "exactly n agents, chosen with CRN".

### 10. CRN-safe "choose N of these UIDs"

- **Problem:** five Covasim places need "pick exactly N of these agents, optionally weighted, without replacement" and use hand-seeded numpy for it: `test_num` weighted choice (`interventions.py:201`), contact-tracing capacity (`interventions.py:271`), `vaccinate_num` (`interventions.py:554`), `clip_edges` (`interventions.py:898`), variant import selection (`immunity.py:129`), plus the seeding above. `ss.choice` (`distributions.py:1676`) chooses among options per agent; it isn't this.
- **Proposal:** a helper such as `ss.choose_uids(uids, n, weights=None)` (or a `Dist` method) that is CRN-stable: e.g. draw one CRN uniform per candidate UID and take the n smallest (weighted: the Efraimidis–Spirakis key `u**(1/w)`). This makes these draws reproducible across scenarios like the rest of Starsim.

### 11. Integer rounding option on distributions

- **Problem:** v3 durations are `lognormal_int` (rounded to whole days). Matching this was needed for parity: it closed peak gaps from −10–13% to −3%. Covasim does it by hand in `COVID._dur` (`covid.py:296`).
- **Proposal:** a `round=True` (or `integer=True`) option on continuous distributions, or `ss.lognorm_ex(..., round=True)`.

## P3

### 12. Extra fields in the infection log

`cv.TransTree` keeps its own log instead of using `ss.infection_log` (`analyzers.py:25`), partly because it needs the variant and the layer of each infection, and because the log is off unless the analyzer is attached in advance (v3 users call `sim.make_transtree()` after a run). Proposal: let a disease add extra fields to the log entry (variant), and make the log cheap enough to be on by default, or allow opting in with a sim par rather than an analyzer.

### 13. Built-in network building blocks

Covasim builds its household/school/work/community ("hybrid") and random layers with v3 code from `population.py` and its own numpy RNG (`network.py:74`). Useful upstream: a static random network with a fixed number of contacts per agent (Poisson), and a clustered/household network (fully connected clusters of a given size distribution), both using CRN. Covasim's layers would then be configurations of these.

### 14. `MultiSim` compatibility hooks

Building `cv.MultiSim` on `ss.MultiSim` should mostly be Covasim-side work, but check whether `ss.MultiSim` can accommodate these without Covasim overriding internals:

- v3 reduced results are Result-like objects with `.values`/`.low`/`.high`, where Starsim's are dicts (`run.py:267`)
- `combine`, `merge`, `split`, `compare`
- serial execution by default

A `reduce` that returns `ss.Result` objects with `low`/`high` attributes would be the key piece.

## Covasim-side fixes (listed for completeness, not Starsim changes)

- `cv.change_beta` calls `float()` on an `ss.probperday` (`interventions.py:832`), which Starsim ≥3.6 intentionally disallows (`ss.Rate.__float__`, `time.py:1689`). Fix in Covasim by working with the rate's value explicitly.
- After the updates above are released, pin `starsim>=<that version>` in `pyproject.toml` (currently unpinned).
