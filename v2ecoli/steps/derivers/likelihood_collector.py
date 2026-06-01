"""LikelihoodCollector — aggregate per-process log-likelihoods (Phase 3).

Reads `listeners.rnap_data.log_likelihood` and
`listeners.ribosome_data.log_likelihood` (emitted by TranscriptInitiation
and PolypeptideInitiation in poisson mode), writes per-process values
and their sum to `listeners.likelihood`.

Serves two purposes:

1. **Aggregate** the per-tick log-likelihood for downstream inference
   tools (ABC-SMC, SBC, etc.) that want a single scalar observable per
   tick instead of fishing it out of multiple per-process listeners.

2. **Pin** the per-process log-likelihood scalar writes in the merged
   state. The pbg merger prunes scalar listener fields with no
   downstream consumer (task #14); the collector's input declaration
   on those paths IS the consumer that keeps them alive. Replaces the
   sprint-1 workaround that declared `log_likelihood` as an input on
   the `RnapData` listener step (a misuse of that step's interface).
"""
from __future__ import annotations

from v2ecoli.library.ecoli_step import EcoliStep as Step


NAME = "likelihood_collector"
TOPOLOGY = {
    "listeners": ("listeners",),
}


class LikelihoodCollector(Step):
    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "tick_s": {"_default": 1.0},
    }

    def initialize(self, config):
        self.tick_s = float(self.parameters.get("tick_s", 1.0))

    def inputs(self):
        return {
            "listeners": {
                "rnap_data": {
                    "log_likelihood": {
                        "_type": "overwrite[float]", "_default": 0.0},
                },
                "ribosome_data": {
                    "log_likelihood": {
                        "_type": "overwrite[float]", "_default": 0.0},
                },
            },
        }

    def outputs(self):
        return {
            "listeners": {
                "likelihood": {
                    "transcript_init": {
                        "_type": "overwrite[float]", "_default": 0.0},
                    "polypeptide_init": {
                        "_type": "overwrite[float]", "_default": 0.0},
                    "total": {
                        "_type": "overwrite[float]", "_default": 0.0},
                },
            },
        }

    def update(self, states, interval=None):
        ti = float(
            states["listeners"]["rnap_data"].get("log_likelihood", 0.0))
        pi = float(
            states["listeners"]["ribosome_data"].get("log_likelihood", 0.0))
        return {
            "listeners": {
                "likelihood": {
                    "transcript_init": ti,
                    "polypeptide_init": pi,
                    "total": ti + pi,
                }
            }
        }
