'''
Backwards compatibility with Covasim v3.

Covasim v4 is built on Starsim, which names some things differently (e.g. a module's
``step()`` rather than v3's ``apply(sim)``). The shims here let v3 scripts run unchanged.
They are kept together in this file so the rest of the code can use the Starsim
conventions directly. Each one notes the v3 behavior it preserves.
'''

import functools as ft
import inspect
import numpy as np
import sciris as sc
import starsim as ss
from . import parameters as cvpar
from . import network as cvnet

__all__ = ['V3Module', 'V3Sim', 'V3Summary', 'V3People', 'V3MultiSim', 'V3Scenarios']


def _with_default_sim(func):
    ''' Let a v3-style method with a required "sim" argument be called without it, as Starsim does '''
    @ft.wraps(func)
    def wrapper(self, sim=None, *args, **kwargs):
        return func(self, self.sim if sim is None else sim, *args, **kwargs)
    return wrapper


class V3Module:
    '''
    Mixin for Covasim interventions and analyzers that supports the v3 module methods.

    In v3, custom interventions and analyzers defined ``apply(self, sim)`` (called on
    each timestep), and optionally ``initialize(self, sim)`` and ``finalize(self, sim)``.
    In Starsim these are ``step()``, ``init_post()``, and ``finalize()``. A subclass can
    define either set.
    '''

    def __init__(self, *args, do_plot=None, show_label=None, line_args=None, **kwargs):
        ''' v3: accept the plotting kwargs, which are used by sim.plot() '''
        super().__init__(*args, **kwargs)
        self.do_plot = do_plot
        self.show_label = show_label
        self.line_args = line_args
        return

    def __init_subclass__(cls, **kwargs):
        ''' v3: finalize(self, sim) -- Starsim calls finalize() with no arguments, so supply the sim '''
        super().__init_subclass__(**kwargs)
        func = cls.__dict__.get('finalize')
        if func is not None:
            sim_arg = inspect.signature(func).parameters.get('sim')
            if sim_arg is not None and sim_arg.default is inspect.Parameter.empty:
                cls.finalize = _with_default_sim(func)
        return

    def init_post(self):
        ''' v3: initialize(self, sim) is called once the sim is initialized '''
        super().init_post()
        if type(self).initialize is not V3Module.initialize:
            self.initialize(self.sim)
        return

    def initialize(self, sim=None):
        ''' v3: override to initialize the module once the sim is available (use init_post() in v4) '''
        pass

    def step(self):
        ''' v3: apply(self, sim) is called on each timestep '''
        if type(self).apply is not V3Module.apply:
            return self.apply(self.sim)
        return

    def apply(self, sim):
        ''' v3: override to apply the module on each timestep (use step() in v4) '''
        pass

    def finalize(self, sim=None):
        ''' v3: finalize(sim) accepts the sim (use finalize() in v4) '''
        return super().finalize()



class V3Sim:
    '''
    Mixin for cv.Sim that supports the v3 dict-style parameter access, e.g. ``sim['beta'] = 0.02``.

    v3 kept all parameters in one dict. In v4, the sim-level parameters are in ``sim.pars`` (with
    Starsim names, e.g. ``n_agents`` rather than ``pop_size``), and the disease parameters are in
    ``sim.diseases.covid.pars``. The v3 keys below are translated; any other key is looked up in the
    COVID parameters, and then as a sim attribute (the Starsim behavior, e.g. ``sim['diseases']``).
    '''

    # Parameters that determine how the sim is built, so can only be changed before it is initialized
    _v3_init_keys = ['pop_size', 'pop_type', 'pop_infected', 'n_days', 'start_day', 'end_day', 'rand_seed', 'pop_scale', 'use_waning']

    def _v3_covid(self):
        ''' The COVID module: the one in the sim after initialization, else the one in the pars '''
        if self.initialized:
            return self.diseases.get('covid')
        diseases = self.pars.diseases
        return diseases if hasattr(diseases, 'pars') else None # cv.Sim stores a single COVID module

    def _v3_networks(self):
        ''' The networks, before or after initialization '''
        return list(self.networks.values()) if self.initialized else sc.tolist(self.pars.networks)

    def __getitem__(self, key):
        ''' v3: sim['key'] returns the parameter '''
        covid = self._v3_covid()
        if key == 'pop_size':
            return self.pars.n_agents
        elif key == 'pop_type':
            return self.pop_type
        elif key == 'pop_infected':
            return covid.pars.init_prev
        elif key == 'n_days':
            return int(self.pars.dur.value)
        elif key == 'start_day':
            return sc.date(self.pars.start, as_date=False)
        elif key == 'end_day':
            return sc.datedelta(self['start_day'], days=self['n_days'])
        elif key == 'rand_seed':
            return self.pars.rand_seed
        elif key == 'pop_scale': # Before initialization, Starsim may not have calculated this yet
            if self.pars.pop_scale is not None:
                return self.pars.pop_scale
            else:
                return self.pars.total_pop/self.pars.n_agents if self.pars.total_pop else 1.0
        elif key == 'verbose':
            return self.pars.verbose
        elif key == 'beta': # Stored as a rate; v3 used the daily probability
            beta = covid.pars.beta
            return beta if isinstance(beta, dict) else ss.probperday(beta).value
        elif key == 'contacts':
            return {net.name: net.n_contacts for net in self._v3_networks()}
        elif key == 'n_variants':
            return covid.nv
        elif key in ['variant_map', 'variant_pars']:
            return getattr(covid, key)
        elif key == 'prognoses': # Only stored if customized
            return covid.pars.prognoses if covid.pars.prognoses is not None else cvpar.get_prognoses(by_age=covid.pars.prog_by_age)
        elif key in ['interventions', 'analyzers']:
            return list(getattr(self, key).values()) if self.initialized else sc.tolist(self.pars[key])
        elif covid is not None and key in covid.pars:
            return covid.pars[key]
        else:
            return getattr(self, key)

    def __setitem__(self, key, value):
        ''' v3: sim['key'] = value sets the parameter '''
        covid = self._v3_covid()
        if key in self._v3_init_keys and self.initialized:
            errormsg = f'Cannot set "{key}" after the sim has been initialized; please set it when creating the sim'
            raise RuntimeError(errormsg)

        if key == 'pop_size':
            self.pars.n_agents = value
        elif key == 'pop_infected':
            covid.pars.init_prev = int(value)
        elif key == 'n_days':
            self.pars.dur = ss.days(value)
        elif key == 'start_day':
            n_days = self['n_days']
            self.pars.start = ss.date(value)
            self['n_days'] = n_days
        elif key == 'end_day':
            self.pars.dur = ss.days(sc.daydiff(self['start_day'], value))
        elif key == 'rand_seed':
            self.pars.rand_seed = value
        elif key == 'pop_scale':
            self.pars.pop_scale = value
        elif key == 'verbose':
            self.pars.verbose = value
        elif key in ['interventions', 'analyzers']:
            if self.initialized:
                errormsg = f'Cannot set "{key}" after the sim has been initialized; please set them when creating the sim'
                raise RuntimeError(errormsg)
            self.pars[key] = value
        elif key == 'beta':
            covid.pars.beta = value if isinstance(value, (dict, ss.Rate)) else ss.probperday(value)
        elif key in ['pop_type', 'contacts']: # Recreate the networks (and for pop_type, the per-layer parameters)
            if self.initialized:
                errormsg = f'Cannot set "{key}" after the sim has been initialized; please set it when creating the sim'
                raise RuntimeError(errormsg)
            pop_type = value if key == 'pop_type' else self.pop_type
            contacts = value if key == 'contacts' else None
            if sc.isnumber(contacts): # A single value for all layers
                contacts = {lkey:contacts for lkey in self.layer_keys()}
            if key == 'pop_type':
                layer_pars = dict(pop_type=pop_type)
                cvpar.reset_layer_pars(layer_pars)
                covid.pars.update({k:layer_pars[k] for k in ['beta_layer', 'iso_factor', 'quar_factor']})
            self.pop_type = pop_type
            self.pars.networks = cvnet.make_networks(pop_type, contacts=contacts)
        elif key in ['use_waning', 'n_variants', 'variant_map', 'variant_pars']:
            errormsg = f'Cannot set "{key}" on an existing sim; please set it when creating the sim'
            raise ValueError(errormsg)
        elif covid is not None and key in covid.pars:
            covid.pars[key] = value
        elif hasattr(self, key): # A Starsim attribute
            setattr(self, key, value)
        else: # As in v3, raise an error for an unrecognized parameter (e.g. a typo)
            keys = list(cvpar.make_pars().keys()) + (list(covid.pars.keys()) if covid is not None else [])
            errormsg = f'Parameter "{key}" not recognized; the parameters are: {sc.strjoin(list(dict.fromkeys(keys)))}'
            raise KeyError(errormsg)
        return

    # v3 properties and methods

    @property
    def npts(self):
        ''' v3: the number of timepoints '''
        return self.t.npts if self.initialized else self['n_days'] + 1

    @property
    def tvec(self):
        ''' v3: the vector of day indices '''
        return np.arange(self.npts)

    @property
    def datevec(self):
        ''' v3: the vector of dates '''
        return np.array(sc.daterange(self['start_day'], self['end_day'], as_date=True))

    @property
    def n(self):
        ''' v3: the number of agents '''
        return len(self.people) if self.initialized else self['pop_size']

    @property
    def scaled_pop_size(self):
        ''' v3: the number of agents times the population scale factor '''
        return self['pop_size']*self['pop_scale']

    def layer_keys(self):
        ''' v3: the keys of the contact layers (i.e., networks) '''
        return [net.name for net in self._v3_networks()]

    def result_keys(self, which='main'):
        ''' v3: the keys of the COVID results; which can be 'main', 'variant', or 'all' '''
        choices = ['main', 'variant', 'all']
        if which not in choices:
            errormsg = f'Choice "{which}" not available; choices are: {sc.strjoin(choices)}'
            raise ValueError(errormsg)
        if not self.initialized: # There are no results yet
            return []
        results = self.diseases.covid.results
        keys = []
        if which in ['main', 'all']:
            keys += [key for key,res in results.items() if isinstance(res, ss.Result)]
        if which in ['variant', 'all']:
            keys += [key for key,res in results['variant'].items() if isinstance(res, ss.Result)]
        return keys

    def update_pars(self, pars=None, **kwargs):
        ''' v3: update the parameters from a dict and/or keyword arguments '''
        for key,val in sc.mergedicts(pars, kwargs).items():
            self[key] = val
        return

    def set_seed(self, seed=-1):
        ''' v3: set the seed from the stored or supplied value (also seeds NumPy's global random number generator, as in v3) '''
        if seed != -1:
            self['rand_seed'] = seed
        np.random.seed(self['rand_seed'])
        return

    def step(self):
        ''' v3: run a single timestep (use run_one_step() in v4) '''
        return self.run_one_step()

    def run(self, do_plot=False, until=None, restore_pars=None, reset_seed=None, verbose=None, **kwargs):
        '''
        v3: run the sim, optionally plotting, and stopping early if the timelimit or stopping_func is reached.
        As in v3, ``until`` is the day to run until (not including that day). restore_pars and reset_seed are
        ignored, since v4 uses separate random number streams.
        '''
        # Convert v3 "until" (the next timestep to run) to Starsim's (the last date to run)
        if until is not None:
            if not self.initialized:
                self.init()
            until = self.day(until)
            if until > self.npts:
                errormsg = f'Requested to run until t={until} but the simulation end is t={self.npts}'
                raise ss.AlreadyRunError(errormsg)
            if self.ti >= until or self.complete:
                errormsg = f'Simulation is currently at t={self.ti}, requested to run until t={until} which has already been reached'
                raise ss.AlreadyRunError(errormsg)

        timelimit = getattr(self, 'timelimit', None)
        stopping_func = getattr(self, 'stopping_func', None)
        if timelimit is None and stopping_func is None:
            super().run(until=None if until is None else self.t.timevec[until-1], verbose=verbose, **kwargs)
        else: # Run one step at a time to check whether to stop; as in v3, the sim isn't finalized if it stops early
            if not self.initialized:
                self.init()
            T = sc.timer()
            until = self.npts if until is None else until
            while self.ti < until and not self.complete:
                super().run(until=self.t.timevec[self.ti], verbose=verbose, **kwargs)
                if timelimit is not None and T.toc(output=True) > timelimit:
                    sc.printv(f'Time limit ({timelimit} s) exceeded; call sim.finalize() to compute results if desired', 1, self.verbose)
                    break
                if stopping_func is not None and stopping_func(self):
                    sc.printv('Stopping function terminated the simulation; call sim.finalize() to compute results if desired', 1, self.verbose)
                    break
        if do_plot:
            self.plot()
        return self

    def initialize(self, *args, reset=False, **kwargs):
        ''' v3: initialize the sim (use init() in v4); if already initialized, reset=True recreates it from its original state '''
        if self.initialized:
            if not reset:
                return self
            self._restore_orig()
        return self.init(*args, **kwargs)

    def _restore_orig(self):
        ''' Restore the sim to its state before it was initialized (v3 sim.initialize(reset=True)) '''
        if self._orig_sim is None:
            errormsg = 'This sim cannot be reset, since a copy could not be saved before it was initialized (e.g. because it contains objects that cannot be pickled); please create a new sim instead'
            raise RuntimeError(errormsg)
        orig = sc.loadstr(self._orig_sim)
        self.__dict__.clear()
        self.__dict__.update(orig.__dict__)
        for obj in self.__dict__.values(): # Point objects that refer to the sim (e.g. the loop) to this sim rather than the copy
            if getattr(obj, 'sim', None) is orig:
                obj.sim = self
        return

    def shrink(self, skip_attrs=None, in_place=True, inplace=None, **kwargs):
        ''' v3: in_place rather than inplace; skip_attrs is ignored '''
        kwargs.setdefault('die', False) # v3 never raised an error on shrinking
        return super().shrink(inplace=in_place if inplace is None else inplace, **kwargs)

    def summarize(self, how='last', full=None, t=None, sep=None, output=None):
        ''' v3: the summary is the final value of each result, and keys can omit the "covid_" prefix; the other v3 arguments are ignored '''
        summary = super().summarize(how=how)
        self.summary = V3Summary({key:val for key,val in summary.items() if f'covid_{key}' not in summary and sc.isnumber(val)}) # Skip the top-level references to the COVID results, and non-scalar results (e.g. by variant)
        return self.summary

    def compute_summary(self, full=None, t=None, update=True, output=False, require_run=False):
        ''' v3: compute the summary of the results (use summarize() in v4) '''
        if require_run and not self.complete:
            errormsg = 'Simulation not yet run'
            raise RuntimeError(errormsg)
        summary = self.summarize()
        if output:
            return summary
        else:
            sc.pp(summary)
            return

    def export_results(self, for_json=True, filename=None, indent=2, *args, **kwargs):
        ''' v3: convert the COVID results to a dict, optionally saving to JSON '''
        results = self.diseases.covid.results
        output = {key:results[key].values.tolist() if for_json else results[key].values for key in self.result_keys()}
        output['t'] = self.tvec.tolist() if for_json else self.tvec
        output['date'] = [str(d) for d in self.datevec]
        if filename is not None:
            sc.savejson(filename=filename, obj=output, indent=indent, *args, **kwargs)
        return output


class V3Summary(sc.objdict):
    ''' v3: the summary, with keys such as summary['cum_infections'] as well as the v4 summary['covid_cum_infections'] '''

    def _key(self, key):
        if isinstance(key, str) and not super().__contains__(key) and super().__contains__(f'covid_{key}'):
            key = f'covid_{key}'
        return key

    def __getitem__(self, key, *args, **kwargs):
        return super().__getitem__(self._key(key), *args, **kwargs)

    def __contains__(self, key):
        return super().__contains__(self._key(key))


class V3People:
    '''
    Mixin for cv.People that supports the v3 per-agent attributes and methods.

    In v3, the disease states (e.g. ``people.exposed``) were stored on People; in v4 they are on
    the COVID module (``sim.diseases.covid.exposed``), so they are looked up there. v3 dates (e.g.
    ``people.date_infectious``) are the v4 time indices (``ti_infectious``), since the timestep
    is one day. As in v4, indexing uses UIDs, e.g. ``people.age[cv.true(people.exposed)]``.
    '''

    def _v3_covid(self):
        ''' The COVID module, or None if the sim is not initialized '''
        sim = self.__dict__.get('sim') # __dict__ access avoids re-triggering __getattr__
        diseases = getattr(sim, 'diseases', None) if sim is not None else None
        return diseases.get('covid') if diseases is not None else None

    def __getattr__(self, key):
        ''' v3: states such as people.exposed and people.date_exposed; only called if normal attribute lookup fails '''
        covid = self._v3_covid()
        if covid is not None and not key.startswith('_'):
            if key.startswith('date_') and hasattr(covid, 'ti_' + key[5:]): # e.g. date_exposed -> ti_exposed
                return getattr(covid, 'ti_' + key[5:])
            state = getattr(covid, key, None)
            if isinstance(state, ss.Arr):
                return state
        errormsg = f"'{self.__class__.__name__}' object has no attribute '{key}'"
        raise AttributeError(errormsg)

    @property
    def sex(self):
        ''' v3: sex as an integer array, 0 for female and 1 for male '''
        return self.female.asnew((~self.female).values.astype(int))

    def true(self, key):
        ''' v3: the UIDs of people for whom this state is true '''
        return self[key].true()

    def false(self, key):
        ''' v3: the UIDs of people for whom this state is false '''
        return self[key].false()

    def defined(self, key):
        ''' v3: the UIDs of people for whom this state is not NaN '''
        return self[key].notnan.true()

    def undefined(self, key):
        ''' v3: the UIDs of people for whom this state is NaN '''
        return self[key].isnan.true()

    def count(self, key):
        ''' v3: the number of people for whom this state is true '''
        return np.count_nonzero(self[key])

    def count_not(self, key):
        ''' v3: the number of people for whom this state is false '''
        return len(self[key]) - self.count(key)

    def layer_keys(self):
        ''' v3: the keys of the contact layers (i.e., networks) '''
        return list(self.sim.networks.keys())

    def keys(self):
        ''' v3: the names of all the per-agent states, including the COVID ones '''
        covid = self._v3_covid()
        covid_keys = [state.name for state in covid.state_list] if covid is not None else []
        return list(self.states.keys()) + covid_keys

    def state_keys(self):
        ''' v3: the names of the COVID true/false states '''
        return [state.name for state in self._v3_covid().state_list if isinstance(state, ss.BoolArr)]

    def date_keys(self):
        ''' v3: the names of the COVID dates, e.g. date_exposed '''
        return ['date_' + state.name[3:] if state.name.startswith('ti_') else state.name for state in self._v3_covid().state_list if state.name.startswith(('ti_', 'date_'))]

    @property
    def infection_log(self):
        ''' v3: the transmission log, a list of dicts with keys source, target, date, layer, and variant '''
        from . import analysis as cva # Here to avoid a circular import
        return cva.make_infection_log(self.sim)

    def indices(self):
        ''' v3: the indices (UIDs) of the agents '''
        return self.auids

    def story(self, uid, *args):
        '''
        v3: print a short history of events in the life of the specified individual(s).

        **Example**::

            sim = cv.Sim(pop_type='hybrid', verbose=0).run()
            sim.people.story(12)
        '''
        layer_labels = dict(a='default contact', h='household', s='school', w='workplace', c='community')
        dates = {
            'date_critical'       : 'became critically ill and needed ICU care',
            'date_dead'           : 'died ☹',
            'date_diagnosed'      : 'was diagnosed with COVID',
            'date_end_quarantine' : 'ended quarantine',
            'date_infectious'     : 'became infectious',
            'date_pos_test'       : 'received their positive test result',
            'date_quarantined'    : 'entered quarantine',
            'date_recovered'      : 'recovered',
            'date_severe'         : 'developed severe symptoms and needed hospitalization',
            'date_symptomatic'    : 'became symptomatic',
            'date_tested'         : 'was tested for COVID',
            'date_vaccinated'     : 'was vaccinated against COVID',
        }
        for uid in sc.tolist(uid) + list(args):
            sex = 'female' if self.female[uid] else 'male'
            intro = f'\nThis is the story of {uid}, a {self.age[uid]:.0f} year old {sex}'
            if not self.susceptible[uid] or self.recovered[uid]:
                print(f'{intro}, who had {"asymptomatic" if np.isnan(self.date_symptomatic[uid]) else "symptomatic"} COVID.')
            else:
                print(f'{intro}, who did not contract COVID.')

            total_contacts = 0
            no_contacts = []
            for lkey, net in self.sim.networks.items():
                llabel = layer_labels.get(lkey, f'"{lkey}"')
                n_contacts = np.count_nonzero(net.edges.p1 == uid) + np.count_nonzero(net.edges.p2 == uid)
                total_contacts += n_contacts
                if n_contacts:
                    print(f'{uid} is connected to {n_contacts} people in the {llabel} layer')
                else:
                    no_contacts.append(llabel)
            if len(no_contacts):
                print(f'{uid} has no contacts in the {", ".join(no_contacts)} layer(s)')
            print(f'{uid} has {total_contacts} contacts in total')

            events = []
            for attr, message in dates.items():
                date = getattr(self, attr)[uid]
                if not np.isnan(date):
                    events.append((date, message))
            if len(events):
                for day, event in sorted(events, key=lambda x: x[0]):
                    print(f'On day {day:.0f}, {uid} {event}')
            else:
                print(f'Nothing happened to {uid} during the simulation.')
        return


class V3MultiSim:
    '''
    Mixin for cv.MultiSim that supports the v3 display and export methods, which differ from
    the Starsim ones (e.g. v3's ``msim.summarize()`` printed a description, whereas Starsim's
    computes summary statistics).
    '''

    def result_keys(self):
        ''' v3: the result keys of the base sim '''
        try:
            keys = self.base_sim.result_keys()
        except Exception as E:
            errormsg = f'Could not retrieve result keys since base sim not accessible: {str(E)}'
            raise ValueError(errormsg)
        return keys

    def summarize(self, output=False):
        ''' v3: print a moderate length description of the MultiSim (use show() in v4) '''
        return self.show(output=output)

    def brief(self, output=False):
        ''' v3: print (or return) a one-line description of the MultiSim '''
        string = repr(self)
        if not output:
            print(string)
        else:
            return string

    def disp(self, output=False):
        ''' v3: print (or return) a detailed description of the MultiSim '''
        string = sc.prepr(self)
        if not output:
            print(string)
        else:
            return string

    def to_json(self, *args, **kwargs):
        ''' v3: shortcut for base_sim.to_json(); only available after reduce() or combine() '''
        if self.which is None:
            errormsg = 'JSON export only available for reduced sim; please run msim.mean() or msim.median() first'
            raise RuntimeError(errormsg)
        return self.base_sim.to_json(*args, **kwargs)

    def to_excel(self, *args, **kwargs):
        ''' v3: shortcut for base_sim.to_excel(); only available after reduce() or combine() '''
        if self.which is None:
            errormsg = 'Excel export only available for reduced sim; please run msim.mean() or msim.median() first'
            raise RuntimeError(errormsg)
        return self.base_sim.to_excel(*args, **kwargs)


class V3Scenarios:
    ''' Mixin for cv.Scenarios that supports the v3 dict-style access to the metaparameters, e.g. ``scens['n_runs']`` '''

    def __getitem__(self, key):
        ''' v3: scens['key'] returns the metaparameter '''
        return self.pars[key]

    def __setitem__(self, key, value):
        ''' v3: scens['key'] = value sets the metaparameter '''
        self.pars[key] = value
        return

    def update_pars(self, pars=None, **kwargs):
        ''' v3: update the metaparameters from a dict and/or keyword arguments '''
        self.pars.update(sc.mergedicts(pars, kwargs))
        return
