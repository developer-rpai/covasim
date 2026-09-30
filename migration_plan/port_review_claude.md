# Covasim v4 Starsim port: review and work plan

Review of the `starsim-port` branch (v4.0.0) against Covasim v3.1.9, done 2026-09-29. It covers the plan in `migration_plan/`, the code, the tests, the tutorials, and a head-to-head comparison of v3 and v4 on user scripts. Companion file: `covasim_starsim_updates.md` (changes that belong in Starsim rather than Covasim).

The two goals are (1) minimal friction for v3.1.9 users (upgrade and change zero or very few lines; anything that must change gets a migration script and skill), and (2) maximal leverage of Starsim (a real port with minimal duplication). These are much less in tension than they look: most compat breaks are cheap to shim, and most of the duplication is dead code.

## Decisions (Cliff, 2026-09-29)

1. Dead code goes. Keep v3 for comparison outside the tree (`git archive main | tar -x -C <scratch>/cov-v3`, or `git show main:<path>`), not in the package.
2. All the simplifications below are approved. Remove the milestone scaffolding.
3. v3 baselines get committed (Cliff commits).
4. Starsim can be released at any time: a quick fix in Starsim is much better than a hack in Covasim. Target current Starsim, not 3.3.x.
5. Compat shims are OK in this case (overrides the style guide's dunder/`__getattr__` caution), but keep them in one visible place.
6. Scope: synthpops **no**, dynamic rescaling **yes**, layer API **yes**.
7. Names approved: `cv.COVID`, `cv.Network`, `cv.CrossImmunity`.

## Summary of findings

- **The core is a real port.** `cv.COVID(ss.Infection)`, `cv.CrossImmunity(ss.Connector)`, the interventions and analyzers are real Starsim modules using `define_states`, `define_results`/`ss.Result`, `ss.lognorm_ex`/`ss.bernoulli` CRN draws, `trans_rng`, `compute_transmission`, and `request_death`. `cv.Sim` does not override the run loop.
- **Core dynamics match v3.** With regenerated v3 baselines, all 12 parity gates (M1–M6, random and hybrid) pass on both Starsim 3.3.4 and 3.6.1. But the threshold is |z|<5 for M2–M6, M3 gates only 3 metrics, `cum_tests` is informational, and the gates skip in every normal run because the baselines are gitignored.
- **The shell is largely v3.** `base.py`, `utils.py`, `settings.py`, `defaults.py`, `parameters.py` are byte-identical to v3.1.9 (~4,200 lines, ~42% of the active package), mostly unreachable from `cv.Sim`. Plus 9,884 lines in `covasim/_v2_legacy/` that nothing imports.
- **Backwards compatibility is far from the goal.** 122/217 behavioral probes pass (Starsim 3.6.1), 20/34 v3 examples, 16/60 original v3 tests. The plan's validation bar ("existing covasim test suite passes") was replaced rather than met: the v3 tests were moved to `tests/_legacy/` and excluded from collection.
- **Starsim is unpinned and the port breaks on current Starsim.** It was written against 3.3.x; on 3.6.1 the suite has 8 failures and `cv.change_beta` crashes in every sim.
- **One possible genuine model divergence.** Hybrid + testing + tracing + waning is +23% to +78% cumulative infections vs v3 (z≈5). Not yet confirmed or explained.

## 1. Backwards compatibility

### Measurements

| Suite | v3.1.9 | v4 + Starsim 3.6.1 | v4 + Starsim 3.3.4 |
|---|---|---|---|
| Behavioral probes (217, from v3 tutorials/examples/tests) | 217/217 | 122 | 127 |
| v3 `examples/*.py` (34) | 32 (2 are v3 bugs) | 20 | 25 |
| Original v3 tests (`tests/_legacy`, 60 top-level) | 60/60 | 16 | 17 |
| Tutorials (`docs/tutorials/*.ipynb`), errored cells | 0 | 13 | 10 |

About 6 of the passing v4 probes are silently wrong (they run but have no effect).

### Breaks, ranked by user impact

Key: **(a)** shim in v4, **(b)** mechanical migration rule, **(c)** semantic change to document, **(S)** fix in Starsim (see companion file).

1. **Custom interventions silently do nothing** (verified). A function intervention testing `sim.t == 10` and a `cv.Intervention` subclass defining `apply(self, sim)` both fire 22 times in v3 and 0 times in v4, with only a warning.
   - `sim.t` is an `ss.Timeline`, so `sim.t == 10` is always False and `sim.t < 50` raises TypeError. **(S)**: integer-like comparison on `ss.Timeline`.
   - `cv.Intervention.step()` must call `self.apply(self.sim)` when a subclass defines it, and `initialize(sim)`/`finalize(sim)` must be called (same for analyzers). **(a)**
   - `change_beta(days=callable)` is silently inert (`interventions.py:823`). Must work or raise. **(a)**
2. **`change_beta` crashes on Starsim ≥3.6** (`interventions.py:832`, `float(beta[lk])` on an `ss.probperday`). Also breaks Scenarios. **(a)**: use the rate's value properly.
3. **`sim.people` API.** Missing `contacts` and the whole layer API (`cv.Layer`, `add_layer`, `dynam_layer`), `sex`, `date_*`/`dur_*`, `true`/`false`/`defined`/`count`, `story`, `infect`, `save`, `layer_keys`, `flows`, `infection_log`; `cv.make_people` fails. People arrays shrink as agents die (active-only), breaking v3 per-agent indexing. **(a)** for the state proxies (`date_x` ← `ti_x`), **(a)+(S)** for the layer API (decision 6: port it).
4. **Custom analyzers.** `self.t = []` raises "locked attribute t"; a user module named `a` clashes with the random network `a`; `initialize`/`finalize` hooks not called. **(a)** for hooks; **(S)** or **(c)** for reserved names.
5. **Sim attributes/pars.**
   - Missing: `npts`, `tvec`, `datevec`, `n`, `result_keys()`, `compute_summary`, `update_pars`, `set_seed`, `step`, `export_results`, `scaled_pop_size`, `sim['beta_layer']`, `sim['contacts']`, `sim['variant_map']`, `run(do_plot=, reset_seed=, keep_people=)`, `initialize(reset=True)`. **(a)**
   - `sim.pars['pop_size']` raises KeyError. **(a)**
   - Silently ignored: `sim['n_days'] = 10`, `sim['pop_infected'] = 0` before a run. Must apply or raise. **(a)**
   - Rejected: `dur=`, `prognoses=`, `rescale=`, `rescale_factor=`, `timelimit=`, `version=`. **(a)**; `rescale` is real work (decision 6).
   - `_cv_config` goes stale: `sim.beta`/`sim['beta']` report the construction value after `sim['beta'] = x` or `change_beta` (`sim.py:195-222`). **(a)**
6. **Results and exports.**
   - `to_df()`, `to_json()`, `people.to_df()` crash on the 2D by-variant results. Fixed structurally by §2 item 2.
   - `summary['cum_infections']` needs `covid_` prefix. **(a)**: bare-key aliases for the COVID module.
   - Missing results: `incidence`, `doubling_time`, `rel_test_yield`, `cum_reinfections`, `n_naive`, `n_preinfectious`, `frac_vaccinated`, `pop_symp_protection`; `Result.color`. **(a)**
   - `cum_infections[0]` is 0 in v4 vs `pop_infected` in v3. **(a)**
   - `sim.date(10)` returns a `datetime.date`, v3 returns a string. **(a)**
   - `shrink()` raises; `shrink(in_place=)` vs `inplace=`; `save(keep_people=True)` raises. **(a)**
   - `plot(to_plot={title: [keys]})` fails. **(a)**
7. **MultiSim and Scenarios.**
   - `msim.results['x']` is a `{'best','low','high'}` dict, not a Result with `.values/.low/.high`; `base_sim.results` empty after `mean()`.
   - Missing: `compare`, `save`, `summarize`, `brief`, `disp`, `to_excel`, `split`, `init_sims`; `merge(base=True)` ignored; `run(keep_people=, reduce=)`, `multi_run(iterpars=)`, `single_run(sim, beta=...)` fail.
   - `Scenarios(sim=...)` raises; results nesting inverted (`results[scen]` vs v3 `results[key][scen]['best']`); missing `summarize/save/to_json/to_excel`.
   - Mostly **(a)**, and mostly solved by building on `ss.MultiSim` (§2 item 3).
8. **Intervention/analyzer kwargs.**
   - `test_prob`/`test_num`: `quar_policy`, `subtarget`, `swab_delay`, `ili_prev`, `quar_test` not ported (they raise, which is correct). Port them.
   - Two `vaccinate_prob`/`vaccinate_num` (the booster pattern) fail with duplicate name. **(S)** auto-unique names.
   - `get_intervention(1)` by index fails. **(a)**
   - `snapshot('2020-03-10')`: date-string days silently dropped; keys are `'2020.03.21'` not `'2020-03-21'`. **(a)**
   - `nab_histogram()` requires `days`; `daily_age_stats.plot()` crashes; `cv.daily_stats` missing. **(a)**
   - `make_transtree()` needs the analyzer attached in advance; `cv.TransTree(sim)` fails. **(a)** via `ss.infection_log` being always-on or cheap (see companion file).
9. **Calibration.** `n_trials` → `total_trials`; `data` required even when on the sim. **(a)** kwarg alias.
10. **Missing top-level names:** `AlreadyRunError`, `InterventionDict`, `daily_stats`, `make_metapars`, `plot_compare`, `plot_people`, `plot_result`, `plot_scens`, `plot_sim`, `plotly_*`, `demo`, `immunity.linear_decay`. **(a)**; plotly functions may be (c).
11. **Pickles.** A v3 `.sim` loads and `results` are readable, but methods fail. **(c)** document; not worth more.
12. **Dropped:** synthpops (decision 6), `bin/covasim` CLI. **(c)**

Works well already: construction (dict and kwargs), `sim['beta']` get/set, most built-in interventions and vaccines (named vaccines, vaccine `subtarget`, `prior_immunity`, `historical_wave`, variants), `run(until=)`/resume, basic plotting, save/load, `msim.plot/plot_result/mean/median/combine`, `cv.parallel`, `cv.date/day/daydiff`, `check_version`, `git_info`, `Fit`.

### Numerical parity (5–8 seeds, 20k agents, `num_compare.txt`)

| Scenario | cum_infections v4 vs v3 |
|---|---|
| Default random | −0.9% (z=−0.1) |
| Random + test/trace | −3% |
| Random + interventions | −9% (z=−0.4) |
| Hybrid, no interventions | +24% on ss3.3.4 (z=2.8); +8.5% on ss3.6.1 |
| Hybrid + change_beta only | +0.6% |
| Hybrid + testing only | +5% (z≈8) |
| Hybrid + test + trace | +23% (z≈5.7) |
| Hybrid + test/trace/change_beta, waning on, 120 d | **+78% (z=4.8)** |

The M5 gate (hybrid, test + trace) passes, but with `use_waning=False`. Suspects: isolation/quarantine on hybrid layers under waning (quarantine factors are a single scalar rather than per-layer, M5 Open Q A), or testing interacting with reinfection. Must be diagnosed before release, and a hybrid + test/trace + waning parity gate added.

## 2. Starsim leverage

### Per-module status

| Module | LOC | Status | Notes |
|---|---|---|---|
| `covid.py` | 1111 | real module | `infect()` (L516-618) copies `ss.Infection.infect` to add a variant loop; 3 non-CRN RNG uses; per-variant immunity arrays are plain ndarrays (block births, L1075-1083); 82 milestone comments |
| `sim.py` | 560 | compat + duplication | own `plot` (L426-493), `to_excel`, `day`/`date`, manual by-variant scaling in `finalize` (L356-404, 539-560); defaults re-hard-coded (`_BETA_LAYER`, `_BASE_BETA`) |
| `people.py` | 64 | thin adapter | fine |
| `network.py` | 124 | real + duplication | edges built with v3 `population.py` code and `np.random.default_rng` (L74); `_CONTACTS` duplicates `parameters.reset_layer_pars` |
| `connectors.py` | 142 | real | stale "M3 static form" docstring |
| `immunity.py` | 287 | real | NAb math genuine; `cv.variant` has its own RNG (L129) and is driven from `COVID.step_state` |
| `interventions.py` | 985 | real | 4 non-CRN RNG uses; `_find_contacts` (L20) duplicates `ss.Network.find_contacts`; own start/end-day handling |
| `analysis.py` | 683 | real + compat | `cv.Calibration` is a good thin wrapper; `cv.Fit` a reasonable compat reimplementation; `TransTree` uses its own log instead of `ss.infection_log` |
| `run.py` | 247 | duplication | `MultiSim` is a `sc.prettyobj`, not `ss.MultiSim`; own serial loop, `reduce`, `combine`, `plot`; `Scenarios` own class |
| `base.py` | 1877 | dead (verbatim v3) | `BaseSim`/`BasePeople` never instantiated; `cv.Result`/`cv.ParsObj`/`cv.Layer`/`cv.Contacts` are parallel types |
| `utils.py` | 674 | dead (verbatim v3) | numba/global-RNG helpers; only reached via `population.py` `rng=None` fallbacks |
| `misc.py` | 991 | mixed | date aliases, `compute_gof`, `get_doubling_time`, `git_info`, `load` are legitimate; `migrate` (~180 lines) cannot apply |
| `settings.py` | 626 | mostly legit | numba options and `reload_numba` dead |
| `defaults.py` | 445 | mostly dead | only `default_age_data`, `default_int`, `default_float` used |
| `parameters.py` | 609 | mixed | `get_prognoses`/`get_variant_*`/`get_vaccine_*` used; `make_pars`/`reset_layer_pars` dead: a second source of truth |
| `population.py` | 452 | mixed | contact builders used; `make_people`, `make_randpop`, `validate_popdict`, `make_synthpop` (~250 lines) dead and broken |
| `_v2_legacy/` | 9884 | dead | not imported |

Rough split of the 9,972 active lines: ~25% genuine model logic, ~19% compat shims, ~6% duplication of Starsim, ~35% dead. For comparison, ported HPVsim is ~5.9k active lines with no `base.py` and a 188-line `sim.py`.

### Simplifications (approved)

1. **Delete dead code:** `covasim/_v2_legacy/`, `tests/_legacy/` (after its tests are moved back into the suite, see §4), `base.py` except what the compat layer needs, `utils.py` except any public `cv.true`-style helpers users rely on, `population.make_people/make_randpop/validate_popdict/make_synthpop` and `rng=None` branches, `misc.migrate*` and BaseSim branches of `load`, numba settings and `reload_numba`, unused `defaults.*` lists, unused `tt` at `sim.py:305`. Repoint the ~5 comments that cite `_v2_legacy` line numbers.
2. **By-variant results as 1D results** (e.g. `new_infections_by_variant` → `new_infections_wild`, `new_infections_alpha`, …, grouped for the `sim.results['variant'][key]` compat view), or 2D support upstream (companion file). Removes the manual scaling in `sim.finalize`, the custom `sim.plot`/`to_excel`, the custom MultiSim reduce, and fixes the `to_df`/`to_json` crash.
3. **`cv.MultiSim(ss.MultiSim)`** and `cv.Scenarios` built on it, keeping only the v3 argument/result aliases. `cv.parallel` → `ss.parallel`.
4. **CRN everywhere.** Replace the nine hand-seeded streams (each behind a copy-pasted `try: int(self.sim.pars.rand_seed)` block):
   - `covid.py:274` `beta_dist` → `ss.nbinom`
   - `covid.py:754` `n_imports` → `ss.poisson` + CRN selection
   - `covid.py:1107` exact-count seeding → upstream exact-count `init_prev`
   - `network.py:74` edge construction → CRN dists
   - `immunity.py:129` variant import selection
   - `interventions.py:201` `test_num` weighted choice
   - `interventions.py:271` tracing capacity
   - `interventions.py:554` `vaccinate_num` sequence
   - `interventions.py:898` `clip_edges`

   Several need the upstream "choose N UIDs" helper.
5. **Stop copying `ss.Infection.infect`** (`covid.py:516-618`): use the upstream hook (companion file).
6. **One source of truth for defaults.** Make `parameters.py` feed `COVID.define_pars`, the layer betas and contact counts (or the reverse), and delete the other copies.
7. **Growth-aware per-variant immunity** (`sus_imm`/`symp_imm`/`sev_imm` as `ss.FloatArr`s, one per variant, or an upstream 2D state). Unblocks births.
8. **Smaller:** TransTree on `ss.infection_log`; `_find_contacts` → `ss.Network.find_contacts`; `sim.day`/`sim.date` via the Timeline/`ss.date`; `get_interventions`/`get_analyzers` as thin wrappers over Starsim module lookup; `export_pars`/`to_excel` via `ss.Sim.to_json`/`to_df`; `age_histogram`/`daily_age_stats` could use `ss.dynamics_by_age`. Interventions on `ss.BaseTest`/`ss.BaseVaccination`/products only if it simplifies without changing behavior.
9. **Remove milestone scaffolding:** ~145 "M1"…"M10" comments (82 in `covid.py`), 19 "byte-identical" notes, `__init__.py` docstring ("M1 surface", "BaseSim/BasePeople dormant"), error messages "not supported in M1" (`sim.py:87`, `network.py:115`), the `NOTES_FOR_CLIFF` reference (`interventions.py:824`), lazy in-function imports used to dodge cycles (`covid.py:181, 254, 363, 1088`; `immunity.py:130`). Rename `test_mN_parity.py`/`anchor_mN.py` by topic (e.g. `test_parity_natural_history.py`, `test_parity_testing.py`).

### Compat layer

Per decision 5, compat shims are fine, but they should live in one place so the Starsim-native code stays readable. Proposal: a `covasim/compat.py` holding the v3 aliases (properties, `__getitem__`/`__setitem__` routing, kwarg renames, result/summary aliases, `apply()`/`initialize()` dispatch), mixed into or called from `Sim`, `People`, `MultiSim`, `Scenarios`, `Intervention`, `Analyzer`. Each shim gets a one-line comment naming the v3 behavior it preserves. This also makes the list of shims easy to audit and, eventually, to deprecate.

## 3. Plan and tests

### Milestones

| M | Delivered | Gaps |
|---|---|---|
| M0 | regression harness, parity helper, compare CLI | own anchor gate always skipped |
| M1 | `cv.Network`, random + hybrid networks, basic `cv.COVID` | contact-structure test is a stub (`test_network.py:118` raises `NotImplementedError`) |
| M2 | full prognosis tree, viral load, `beta_dist`, static `pop_scale` | bed-capacity feedback untested |
| M3 | single module with variant axis, `cv.CrossImmunity`, `cv.variant` | only 3 metrics hard-gated |
| M4 | NAb engine | births unsupported (`covid.py:1077`) |
| M5 | test_prob/test_num, tracing, quarantine/isolation | quarantine/isolation factors scalar, not per-layer; missing test kwargs |
| M6 | vaccines, `simple_vaccine`, historical immunity | historical NAbs not cross-checked vs v3 |
| M7 | `cv.Fit`, `cv.Calibration` | "reproduces a v3 fit" never done |
| M8 | MultiSim, Scenarios, parallel | not built on `ss.MultiSim`; `noise`/`iterpars` |
| M9 | analyzers, TransTree, plots | no plan/spec; synthpops not ported (now dropped); no TransTree graph |
| M10 | migration guide, version, baselines, save/load | no plan/spec; legacy not deleted; no docs rebuild; no tag/release |

Items the plan marked "port" that were moved to "not yet ported" without sign-off: synthpops (now dropped by decision), dynamic rescaling (now: port), layer API (now: port), TransTree graph.

### Tests

- v4 suite on Starsim 3.6.1: 139 passed, 8 failed, 17 skipped. On 3.3.4: 147 passed, 17 skipped.
- 8 failures: 5 from `change_beta`'s `float(probperday)` (`interventions.py:832`), plus `test_nv1_by_variant_equals_aggregate` and two vaccination-subtarget tests in `test_unported_features.py`.
- All 12 v3↔v4 parity gates skip without baselines (gitignored); the M0 gate always skips.
- The original v3 tests (75 functions in `tests/_legacy/`) are excluded by `conftest.py`; against v4, 58 fail.

## 4. Work plan

Order chosen so silent failures and environment breakage go first, and the migration catalog falls out of the work rather than being written from memory.

1. **Current Starsim.** Fix `change_beta` and the other 3.6 breakages; set `starsim>=<new version>` once the upstream changes in the companion file are released.
2. **Restore the v3 tests as the acceptance bar.** Move `tests/_legacy/*` back into `tests/` (renaming where they collide with the new tests). For every failing test, either (a) add a shim, (b) change the test and record the change as a migration rule in `migration/rules.md` (v3 code → v4 code, one rule per change), or (c) skip with a documented reason (synthpops, pickles). This makes the rules list the specification for the migration script and skill.
3. **Fix silent failures:** `apply()`/`initialize()`/`finalize()` dispatch on interventions and analyzers, integer-like `sim.t` (upstream), callable `days`, ignored `sim[...] =` sets, dropped `snapshot` date strings, stale `_cv_config`.
4. **Commit the v3 baselines** (Cliff) so the parity gates run in CI. Add a hybrid + test/trace + waning gate, then diagnose the hybrid divergence.
5. **Simplifications** in §2 (delete dead code first, then 1D variant results, `ss.MultiSim`, CRN, `infect` hook, defaults, per-variant arrays, scaffolding).
6. **Scope items:** dynamic rescaling, the layer API (`people.contacts`, `cv.Layer`, `add_layer`, `dynam_layer`), missing test kwargs (`quar_policy`, `subtarget`, `swab_delay`, `ili_prev`, `quar_test`), `cv.daily_stats`, missing results, `cv.plot_*` functions.
7. **Probe battery as a permanent test.** Turn the 217-probe v3 script battery (scratch `compat/probes.py`) into `tests/test_v3_compat.py`: each probe must pass or be listed in `migration/rules.md`.
8. **Migration tooling.** From `migration/rules.md`: a migration script (`covasim.migrate` or `python -m covasim.migrate script.py`) that applies the mechanical rules and flags the rest, plus a Claude skill (e.g. in the starsim-ai plugin) that covers the judgment calls. Rewrite `docs/migration.md` from the rules (and unwrap it; it is hard-wrapped).
9. **Release:** delete `migration_plan/`'s transitional notes or move to an archive, docs/tutorial rebuild (all 11 tutorials must run clean), CHANGELOG, then Cliff tags and releases.

## 5. Open questions

- `sim.t` design: exact semantics of the upstream Timeline comparison (see companion file, item 1).
- Reserved module names: should user analyzers/interventions be able to use `t`, and should a user module named `a` be allowed to coexist with the random network `a` (upstream auto-renaming, or a v4 migration rule)?
- People arrays: should `sim.people.age` etc. in v3-compat mode return all agents (including dead), matching v3 indexing, or active agents only?
- Hybrid divergence: root cause and whether it is a v4 bug or an accepted difference.
- Whether `migration_plan/` stays in the repo after release.

## Evidence

Scratch outputs from the review (session scratchpad, not in the repo): `compat/` (API diff, 217-probe battery and results, example/test runs, numerical comparisons), `plan/` (test runs on both Starsim versions, executed notebooks, regenerated v3 baselines), `leverage/`. To run v3 for comparison: `git archive main | tar -x -C <dir>`, then `PYTHONPATH=<dir> python script.py` (setting the cwd alone doesn't shadow the editable install when running a script).
