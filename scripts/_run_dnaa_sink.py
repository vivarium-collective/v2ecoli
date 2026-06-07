# Task-2 sink runner: enable the dnaa_sink feature, then delegate to the
# proven multigen parquet runner. The feature swaps the read-only
# dnaa_box_binding_listener for dnaa_box_binding_sink (partitions the bulk
# DnaA pool). Not committed — a run driver only.
import sys, os
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from v2ecoli.composites.baseline import enable_features
enable_features('dnaa_sink')
print('>>> dnaa_sink feature ENABLED (bulk DnaA pool partitioned)')
import runpy
sys.argv=['run_condition_multigen_parquet.py']+sys.argv[1:]
runpy.run_path(os.path.join(HERE,'run_condition_multigen_parquet.py'), run_name='__main__')
