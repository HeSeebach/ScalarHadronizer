# %%
from scalarhadronizer import ScalarHadronizer

SH=ScalarHadronizer()
m=2000
decay_graph=SH.simulateDecay(m)
SH.plot_hist(decay_graph,m)
fs=SH.get_most_common_final_states(decay_graph)
SH.plot_from_final_state(decay_graph,list(fs.keys())[0])