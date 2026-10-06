"""
Defines the People class.
"""
import numpy as np
import starsim as ss

from . import defaults as cvd
from . import compat as cvc
from . import plotting as cvplt

__all__ = ['People']


def convert_age_data(age_data):
    """
    Convert Covasim's age data, a table of (minimum age, maximum age, fraction) rows (e.g. from
    ``cv.data.get_age_distribution()``), to the (lower edge, value) rows that ``ss.People`` expects.

    ss.People reads the ages as lower bin edges, so the upper edge of the last bin is added as
    a final row with a value of 0. As in v3, ages are uniform within [age_min, age_max+1).
    """
    d = np.asarray(age_data, dtype=float)
    return np.vstack([d[:, [0, 2]], [d[-1, 1] + 1, 0]])


class People(cvc.V3People, ss.People):
    """
    The people in the simulation: their ages and sexes, and (once in a sim) their disease states.

    The disease states (e.g. ``people.exposed``) are stored on the COVID module
    (``sim.diseases.covid``), but can be read from the people as in v3. Dates such as
    ``people.date_infectious`` are the time indices (e.g. ``ti_infectious``), since the timestep
    is one day. The per-agent arrays only include agents who are alive, except for
    ``people.dead`` and ``people.date_dead``, which cover every agent ever created.

    Args:
        n_agents (int): the number of agents
        age_data (array/dataframe): the age distribution, either as Covasim's (minimum age, maximum age, fraction) rows (e.g. from ``cv.data.get_age_distribution()``) or as Starsim's (lower edge, value) rows; defaults to Covasim's default age distribution
        kwargs (dict): passed to ``ss.People`` (e.g. ``extra_states``)

    **Example**::

        people = cv.People(1000, age_data=cv.data.get_age_distribution('nigeria'))
        sim = cv.Sim(people=people, pop_size=1000)
    """

    def __init__(self, n_agents, age_data=None, **kwargs):
        if age_data is None:
            age_data = cvd.default_age_data
        if (np.ndim(age_data) == 2) and (np.shape(age_data)[1] == 3): # Covasim's format, so convert it to Starsim's
            age_data = convert_age_data(age_data)
        super().__init__(n_agents, age_data=age_data, **kwargs)
        return

    def plot(self, *args, **kwargs):
        """Plot statistics of the population -- age distribution, numbers of contacts, and overall weight of contacts (number of contacts multiplied by beta per layer); see cv.plot_people()"""
        return cvplt.plot_people(self, *args, **kwargs)
