from plot import *

dataBasic = read_bench('out/BasicEigen.csv')
dataDecomp = read_bench('out/Decomposition.csv')
# Copy
plot_curves(dataBasic, '*Copy*', title = 'Copy timing', logy=True, filename = 'out/basic_copy')
plot_relative_curves(dataBasic, '*Copy*', baseline = 'BM_Copy_MatrixXd', title = 'Copy relative to simple copy', filename = 'out/basic_copy_vsCopy')
# Add
plot_curves(dataBasic, '*Add*', title = 'Add timing', logy=True, filename = 'out/basic_add')
plot_relative_curves(dataBasic, '*Add*', baseline = 'BM_Copy_MatrixXd', title = 'Add relative to simple copy', filename = 'out/basic_add_vsCopy')
# A*x
plot_curves(dataBasic, '*Mult_VectorXd*', title = 'Mat-vec mult', logy=True, filename = 'out/basic_multmv')
plot_relative_curves(dataBasic, '*Mult_VectorXd*', baseline = 'BM_Copy_MatrixXd', title = 'Mat-vec mult relative to simple copy', filename = 'out/basic_multmv_vsCopy')
plot_relative_curves(dataBasic, '*Mult_VectorXd*', baseline = 'BM_Mult_VectorXd', title = 'Mat-vec mult relative to simple mat-vec mult', filename = 'out/basic_multmv_vsMult')
# A*B
plot_curves(dataBasic, '*Mult_MatrixXd*', title = 'Mat-mat mult', logy=True, filename = 'out/basic_multmm')
plot_relative_curves(dataBasic, '*Mult_MatrixXd*', baseline = 'BM_Copy_MatrixXd', title = 'Mat-mat mult relative to simple copy', filename = 'out/basic_multmm_vsCopy')
plot_relative_curves(dataBasic, '*Mult_MatrixXd*', baseline = 'BM_Mult_MatrixXd', title = 'Mat-mat mult relative to simple mat-mat mult', filename = 'out/basic_multmm_vsMult')

#Decomposition
plot_curves(dataDecomp, '*Decomposition*', title = 'Decomposition', logy=True, filename = 'out/decomp')
plot_relative_curves(dataDecomp, '*Decomposition*', baseline = 'BM_Copy_MatrixXd', title = 'Decomposition relative to simple copy', filename = 'out/decomp_vsCopy')
plot_relative_curves(dataDecomp, '*Decomposition*', baseline = 'BM_Mult_MatrixXd', title = 'Decomposition relative to simple mat-mat mult', filename = 'out/decomp_vsMult')
