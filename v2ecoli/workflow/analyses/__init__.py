"""Native ports of vEcoli DuckDB/sim_data analyses (Analysis subclasses).

Importing this package registers every ported analysis into ANALYSIS_REGISTRY.
"""

from v2ecoli.workflow.analyses import ptools_rna  # noqa: F401
from v2ecoli.workflow.analyses import ptools_rxns  # noqa: F401
from v2ecoli.workflow.analyses import ptools_proteins  # noqa: F401
from v2ecoli.workflow.analyses import mass_fraction_voronoi  # noqa: F401
from v2ecoli.workflow.analyses import central_carbon_metabolism_scatter  # noqa: F401
from v2ecoli.workflow.analyses import ptools_multiscale  # noqa: F401
from v2ecoli.workflow.analyses import ptools_overview  # noqa: F401
from v2ecoli.workflow.analyses import dummy  # noqa: F401
from v2ecoli.workflow.analyses import mass_fraction_summary_view  # noqa: F401
from v2ecoli.workflow.analyses import replication  # noqa: F401
from v2ecoli.workflow.analyses import cell_mass  # noqa: F401
from v2ecoli.workflow.analyses import doubling_time_hist  # noqa: F401
from v2ecoli.workflow.analyses import doubling_time_line  # noqa: F401
from v2ecoli.workflow.analyses import ribosome_production  # noqa: F401
from v2ecoli.workflow.analyses import ribosome_components  # noqa: F401
from v2ecoli.workflow.analyses import ribosome_usage  # noqa: F401
from v2ecoli.workflow.analyses import new_gene_counts  # noqa: F401
from v2ecoli.workflow.analyses import average_monomer_counts  # noqa: F401
from v2ecoli.workflow.analyses import subgenerational_expression_table  # noqa: F401
from v2ecoli.workflow.analyses import protein_counts_validation  # noqa: F401

# The cd1 omics suite — multiseed TSV tables (fluxomics / proteomics /
# transcriptomics / metabolomics / exchange fluxes / higher-order properties).
# Registering them here is what puts them in a batch run's "applicable"
# analysis set, so a multiseed sweep reproduces the cd1 result tables.
from v2ecoli.workflow.analyses import cd1_exchange_fluxes  # noqa: F401
from v2ecoli.workflow.analyses import cd1_fluxomics  # noqa: F401
from v2ecoli.workflow.analyses import cd1_higher_order_properties  # noqa: F401
from v2ecoli.workflow.analyses import cd1_metabolomics  # noqa: F401
from v2ecoli.workflow.analyses import cd1_proteomics  # noqa: F401
from v2ecoli.workflow.analyses import cd1_transcriptomics  # noqa: F401

# Cross-variant COMPARISON analyses (variant-comparison study, showcase-4):
# overlay / group / delta baseline + variants instead of one-panel-per-variant.
from v2ecoli.workflow.analyses import growth_overlay  # noqa: F401
from v2ecoli.workflow.analyses import doubling_time_grouped  # noqa: F401
from v2ecoli.workflow.analyses import mass_fraction_grouped  # noqa: F401
from v2ecoli.workflow.analyses import replication_overlay  # noqa: F401
from v2ecoli.workflow.analyses import proteome_delta  # noqa: F401
from v2ecoli.workflow.analyses import fba_flux_overlay  # noqa: F401
from v2ecoli.workflow.analyses import regulation_overlay  # noqa: F401
from v2ecoli.workflow.analyses import scorecard  # noqa: F401
