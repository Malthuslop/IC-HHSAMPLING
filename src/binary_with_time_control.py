from pyswarms.discrete import BinaryPSO
import logging

# Import modules
import numpy as np
import multiprocessing as mp

from collections import deque

from pyswarms.backend.operators import compute_pbest, compute_objective_function
from pyswarms.backend.topology import Ring
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from pyswarms.base import DiscreteSwarmOptimizer
from pyswarms.utils.reporter import Reporter
import time
class BinaryPSOWithTimeStopCriteria(BinaryPSO):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        init_pos=None,
        velocity_clamp=None,
        vh_strategy="unmodified",
        ftol=-np.inf,
        ftol_iter=1,
    ):
        super(BinaryPSOWithTimeStopCriteria, self).__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            init_pos=init_pos,
            velocity_clamp=velocity_clamp,
            vh_strategy=vh_strategy,
            ftol=ftol,
            ftol_iter=ftol_iter,
        )
    def optimize(
        self, objective_func, iters, n_processes=None, verbose=True, time_limit = np.inf, **kwargs
    ):
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : function
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int, optional
            number of processes to use for parallel particle evaluation
            Defaut is None with no parallelization.
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for objective function

        Returns
        -------
        tuple
            the local best cost and the local best position among the
            swarm.
        """
        # Starting the timer
        start_time = time.time()
        
        # Apply verbosity
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=log_level,
        )
        # Populate memory of the handlers
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        ftol_history = deque(maxlen=self.ftol_iter)
        for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
            # Compute cost for current position and personal best
            self.swarm.current_cost = compute_objective_function(
                self.swarm, objective_func, pool, **kwargs
            )
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(
                self.swarm
            )
            best_cost_yet_found = np.min(self.swarm.best_cost)
            # Update gbest from neighborhood
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm, p=self.p, k=self.k
            )
            if verbose:
                # Print to console
                self.rep.hook(best_cost=self.swarm.best_cost)
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=np.mean(self.swarm.best_cost),
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on time
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                break
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = (
                np.abs(self.swarm.best_cost - best_cost_yet_found)
                < relative_measure
            )
            if i < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            
            # Perform position velocity update
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh
            )
            self.swarm.position = self._compute_position(self.swarm)
        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()
        ].copy()
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()
        
        return (final_best_cost, final_best_pos, elapsed_time)