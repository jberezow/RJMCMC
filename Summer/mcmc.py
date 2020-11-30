import random
import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine
from pyro.distributions import Uniform
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel


class MH(MCMCKernel):
    """
    Initial implementation of MH MCMC
    """
    def setup(self, model, *args, burn=10, lag=1, samples=100,**kwargs):
        r"""
        Optional method to set up any state required at the start of the
        simulation run.

        :param int warmup_steps: Number of warmup iterations.
        :param \*args: Algorithm specific positional arguments.
        :param \*\*kwargs: Algorithm specific keyword arguments.
        """
        self.model = model
        self.burn = burn
        self.lag = lag
        self.samples = samples
        self.args = args
        self.kwargs = kwargs

    def cleanup(self):
        """
        Optional method to clean up any residual state on termination.
        """
        print("Cleanup not implemented")


    def logging(self):
        """
        Relevant logging information to be printed at regular intervals
        of the MCMC run. Returns `None` by default.

        :return: String containing the diagnostic summary. e.g. acceptance rate
        :rtype: string
        """
        return None


    def diagnostics(self):
        """
        Returns a dict of useful diagnostics after finishing sampling process.
        """
        # NB: should be not None for multiprocessing works
        return {}


    def end_warmup(self):
        """
        Optional method to tell kernel that warm-up phase has been finished.
        """
        pass

    def initial_params(self):
        """
        Returns a dict of initial params (by default, from the prior) to initiate the MCMC run.

        :return: dict of parameter values keyed by their name.
        """
        prototype_samples = {}
        trace = poutine.trace(self.model).get_trace(self.args, self.kwargs)
        for name, node in trace.iter_stochastic_nodes():
            if (node['type'] == 'sample' and node['is_observed'] == False):
                prototype_samples[name] = node["value"].detach()

        return prototype_samples

    def sample(self, params):
        """
        Samples parameters from the posterior distribution, when given existing parameters.

        :param dict params: Current parameter values.
        :param int time_step: Current time step.
        :return: New parameters from the posterior distribution.
        """
        old_model_trace = poutine.trace(self.model)(self.args, self.kwargs)
        traces = []
        t = 0
        i = 0
        while t < self.burn + self.lag * self.samples:
            i += 1
            # q(z' | z)
            new_guide_trace = poutine.block(
                poutine.trace(self.model))(old_model_trace, self.args, self.kwargs)
            # p(x, z')
            new_model_trace = poutine.trace(
                poutine.replay(self.model, new_guide_trace))(self.args, self.kwargs)
            # q(z | z')
            old_guide_trace = poutine.block(
                poutine.trace(
                    poutine.replay(self.guide, old_model_trace)))(new_model_trace,
                                                                  self.args, self.kwargs)
            # p(x, z') q(z' | z) / p(x, z) q(z | z')
            logr = new_model_trace.log_pdf() + new_guide_trace.log_pdf() - \
                old_model_trace.log_pdf() - old_guide_trace.log_pdf()
            rnd = pyro.sample("mh_step_{}".format(i),
                              Uniform(torch.zeros(1), torch.ones(1)))

            if torch.log(rnd).data[0] < logr.data[0]:
                # accept
                t += 1
                old_model_trace = new_model_trace
                if t <= self.burn or (t > self.burn and t % self.lag == 0):
                    yield (new_model_trace, new_model_trace.log_pdf())


    def __call__(self, params):
        """
        Alias for MCMCKernel.sample() method.
        """
        return self.sample(params)

    def _traces(self, *args, **kwargs):
        """
        make trace posterior distribution
        """
        # initialize traces with a draw from the prior
        old_model_trace = poutine.trace(self.model)(*args, **kwargs)
        traces = []
        t = 0
        i = 0
        while t < self.burn + self.lag * self.samples:
            i += 1
            # q(z' | z)
            new_guide_trace = poutine.block(
                poutine.trace(self.model))(old_model_trace, *args, **kwargs)
            # p(x, z')
            new_model_trace = poutine.trace(
                poutine.replay(self.model, new_guide_trace))(*args, **kwargs)
            # q(z | z')
            old_guide_trace = poutine.block(
                poutine.trace(
                    poutine.replay(self.model, old_model_trace)))(new_model_trace,
                                                                  *args, **kwargs)
            # p(x, z') q(z' | z) / p(x, z) q(z | z')
            logr = new_model_trace.log_pdf() + new_guide_trace.log_pdf() - \
                old_model_trace.log_pdf() - old_guide_trace.log_pdf()
            rnd = pyro.sample("mh_step_{}".format(i),
                              Uniform(torch.zeros(1), torch.ones(1)))

            if torch.log(rnd).data[0] < logr.data[0]:
                # accept
                t += 1
                old_model_trace = new_model_trace
                if t <= self.burn or (t > self.burn and t % self.lag == 0):
                    yield (new_model_trace, new_model_trace.log_pdf())