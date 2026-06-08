"""Native ports of vEcoli DuckDB/sim_data analyses (Analysis subclasses).

Importing this package registers every ported analysis into ANALYSIS_REGISTRY.
"""

from v2ecoli.workflow.analyses import ptools_rna  # noqa: F401
from v2ecoli.workflow.analyses import ptools_rxns  # noqa: F401
from v2ecoli.workflow.analyses import ptools_proteins  # noqa: F401
from v2ecoli.workflow.analyses import mass_fraction_voronoi  # noqa: F401
from v2ecoli.workflow.analyses import central_carbon_metabolism_scatter  # noqa: F401
from v2ecoli.workflow.analyses import ptools_multiscale  # noqa: F401
from v2ecoli.workflow.analyses import dummy  # noqa: F401
from v2ecoli.workflow.analyses import mass_fraction_summary_view  # noqa: F401
from v2ecoli.workflow.analyses import replication  # noqa: F401
from v2ecoli.workflow.analyses import cell_mass  # noqa: F401
from v2ecoli.workflow.analyses import doubling_time_hist  # noqa: F401
from v2ecoli.workflow.analyses import doubling_time_line  # noqa: F401
from v2ecoli.workflow.analyses import ribosome_production  # noqa: F401
from v2ecoli.workflow.analyses import ribosome_components  # noqa: F401
from v2ecoli.workflow.analyses import ribosome_usage  # noqa: F401
