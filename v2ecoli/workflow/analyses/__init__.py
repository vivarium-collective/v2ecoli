"""Native ports of vEcoli DuckDB/sim_data analyses (Analysis subclasses).

Importing this package registers every ported analysis into ANALYSIS_REGISTRY.
"""

from v2ecoli.workflow.analyses import ptools_rna  # noqa: F401
from v2ecoli.workflow.analyses import ptools_rxns  # noqa: F401
from v2ecoli.workflow.analyses import ptools_proteins  # noqa: F401
from v2ecoli.workflow.analyses import mass_fraction_voronoi  # noqa: F401
from v2ecoli.workflow.analyses import central_carbon_metabolism_scatter  # noqa: F401
from v2ecoli.workflow.analyses import ptools_multiscale  # noqa: F401
