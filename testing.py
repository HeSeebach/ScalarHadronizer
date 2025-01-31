# %%
from scalarhadronizer import ScalarHadronizer

m=2000
SH=ScalarHadronizer(m,OAM_supression_par=0,up_weight=1,down_weight=1,strange_weight=1e-5)
decay_graph=SH.simulateDecay()
SH.plot_hist(decay_graph)
#fs=SH.get_most_common_final_states(decay_graph)
#SH.plot_from_final_state(decay_graph,list(fs.keys())[0])
# %%
