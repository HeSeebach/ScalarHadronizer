# %%
from particle import Particle 
import numpy as np 
import xml.etree.ElementTree as ET
import itertools
from math import prod
from bs4 import BeautifulSoup
from collections import Counter
import networkx as nx 
import textwrap

class ScalarHadronizer:
    def __init__(self,path_to_decayXML='DECAY.XML'):
        self.all_decaysXML=self.read_decayXML(path_to_decayXML)
        self.set_stable=[111]

    def read_decayXML(self,path_to_decayXML):
        with open(path_to_decayXML, 'r') as f:
            data = f.read()
        Bs_data = BeautifulSoup(data, "xml")
        decayXMLtree = Bs_data.find_all('decay')

        particles_in_evtgen=[x['name'] for x in decayXMLtree]
        all_decaysXML={}
        #append only those with valid monte carlo id
        for p in decayXMLtree:
            try:
                pdgid=Particle.from_evtgen_name(p['name']).pdgid
                all_decaysXML[pdgid]=p
            except:
                None
        return all_decaysXML

    def single_particle_decays(self,pdgid: int):
        branching_ratios={}
        if pdgid not in self.set_stable:
            assert isinstance(pdgid, int), 'Input of get_decays must be a pdgid in integer form'
            try:
                decay=self.all_decaysXML[pdgid]
                channels=decay.find_all('channel')
            except KeyError:
                channels=[]
            total_br=0
            for channel in channels:
                branching_ratio=float(channel['br'])
                if branching_ratio>0.01: 
                    daughters=tuple([int(Particle.from_evtgen_name(x).pdgid) for x in channel['daughters'].split()])
                    branching_ratios[daughters]=branching_ratio
                    total_br+=branching_ratio
            branching_ratios={d:b/total_br for d,b in branching_ratios.items()}
        return branching_ratios
    
    def check_meson_combinations(self,m1,m2,m):
        mesons_to_exclude=[130,310]

        valid=True
        if m1.mass+m2.mass > m: valid=False                                                                 #check mass
        if (m1.invert()!=m1 or m2.invert()!=m2) and m1.invert()!=m2: valid=False                            #check if particle-antiparticle pair
        if m1.invert()==m1 and m2.invert()==m2:        
            if m1.C!=m2.C: valid=False                                                                      #check charge conjugation eigenvalue
            if m1.J==0 and m2.J==0 and m1.P!=m2.P: valid=False
            if ((m1.J==1 and m2.J==0) or (m1.J==0 and m2.J==1)) and m1.P!=(-1)*m2.P: valid=False            #parity must be opposite for l=1
            if ((m1.J==2 and m2.J==0) or (m1.J==0 and m2.J==2)) and m1.P!=m2.P: valid=False                 #parity must be same for l=2
        #if ((m1.pdgid.has_up and not m2.pdgid.has_up) or (m1.pdgid.has_down and not m2.pdgid.has_down) or
        #    (m1.pdgid.has_strange and not m2.pdgid.has_strange) or (m1.pdgid.has_charm and not m2.pdgid.has_charm) or 
        #    (m1.pdgid.has_bottom and not m2.pdgid.has_bottom)): valid=False
        if m1.I!=m2.I: valid=False                                                                          #isospin conservation
        if m1.pdgid in mesons_to_exclude or m2.pdgid in mesons_to_exclude: valid=False
        return valid

    def make_initialMesonPairs(self,scalar_mass):
        mesons_below_threshold=Particle.findall(lambda p: p.mass<scalar_mass and p.pdgid.is_meson==True)
        if not mesons_below_threshold: print('No possible decay products. Maybe mass is too small?')
        meson_pairs={}
        total_weight=0
        for i,m1 in enumerate(mesons_below_threshold):
            for m2 in mesons_below_threshold[i:]:
                if self.check_meson_combinations(m1,m2,scalar_mass):
                    weight=self.initialWeight(m1,m2,scalar_mass)
                    if m1.pdgid<m2.pdgid: meson_pairs[(int(m1.pdgid),int(m2.pdgid))]=weight
                    else: meson_pairs[(int(m2.pdgid),int(m1.pdgid))]=weight
                    total_weight+=weight
        meson_pairs={k:v/total_weight for k,v in meson_pairs.items() if v/total_weight>1e-3}
        new_total_weight=np.sum(list(meson_pairs.values()))
        meson_pairs={k:v/new_total_weight for k,v in meson_pairs.items()}
        return meson_pairs

    def initialWeight(self,m1,m2,scalar_mass,spin_supression_par=1,up_weight=1,down_weight=1,strange_weight=1e-5,charm_weight=1,bottom_weight=1):
        p_restframe=np.sqrt((scalar_mass**2-(m1.mass+m2.mass)**2)*(scalar_mass**2-(m1.mass-m2.mass)**2))/2/scalar_mass

        #spin multiplicity
        if m1.J==m2.J: spin_factor=(2*m1.J+1)*spin_supression_par
        else: spin_factor=(2*m1.J+1)*(2*m2.J+1)

        #quark weights
        quark_weight=1
        if m1.pdgid.has_up: quark_weight*=up_weight
        if m1.pdgid.has_down: quark_weight*=down_weight
        if m1.pdgid.has_strange: quark_weight*=strange_weight
        if m1.pdgid.has_charm: quark_weight*=charm_weight
        if m1.pdgid.has_bottom: quark_weight*=bottom_weight

        if m1.I==0.5 and m2.I==0.5: isospin_factor=0.5
        if m1.I==1 and m2.I==1: isospin_factor=0.33
        else: isospin_factor=1

        weight=p_restframe*quark_weight*spin_factor*isospin_factor
        return weight
        
    def all_decays_of_multiparticle_state(self,mesons):
        all_brs=[]
        all_daughters=[]
        for meson in mesons:
            decays_of_this_meson=self.single_particle_decays(meson)
            if decays_of_this_meson:
                all_daughters.append(list(decays_of_this_meson.keys()))
                all_brs.append(list(decays_of_this_meson.values()))
            else:
                all_daughters.append([[meson]])
                all_brs.append([1])
        all_combinations=[tuple(sorted(list(itertools.chain(*x)))) for x in itertools.product(*all_daughters)] #cartesian product to get all combinations
        br_combinations=[prod(x) for x in itertools.product(*all_brs)]
        all_decays=dict.fromkeys(all_combinations,0)
        for s,b in zip(all_combinations,br_combinations):
            all_decays[s]+=b
        if len(all_decays)==1 and next(iter(all_decays))==mesons: return None
        else:
            br_too_small=[]
            for k,v in all_decays.items():
                if v<1e-2: br_too_small.append(k)
            br_sum=np.sum([v for k,v in all_decays.items() if k not in br_too_small])
            all_decays={k:v/br_sum for k,v in all_decays.items() if k not in br_too_small}
            return all_decays

    def build_decay_graph(self,scalar_mass,decay_graph=None,initial_states=None):
        if decay_graph is None:
            decay_graph=nx.DiGraph()
            initial_states=self.make_initialMesonPairs(scalar_mass)
            decay_graph.add_nodes_from(list(initial_states.keys()))
            print(f'Generated {len(initial_states)} initial meson pairs.')
        all_decays_finished=True
        states_to_decay= [s for s,d in decay_graph.out_degree() if d==0]
        for state in states_to_decay:
            decays_of_this_state=self.all_decays_of_multiparticle_state(state)
            if decays_of_this_state is not None: 
                edges=[(state,x,y) for x,y in decays_of_this_state.items()]
                #for edge in edges:
                #    assert not edge in decay_graph.edges().data('weight'), f'Edge {edge} is overwritten'
                decay_graph.add_weighted_edges_from(edges)
                all_decays_finished=False

        if not all_decays_finished: decay_graph,initial_states=self.build_decay_graph(scalar_mass,decay_graph,initial_states)
        return decay_graph,initial_states

    def buildWeights(self,scalar_mass,decay_graph,initial_states):
        nx.set_node_attributes(decay_graph,0,name='weight')
        nodes=decay_graph.nodes
        for s,w in initial_states.items():
            nodes[s]['weight']=w
        has_weight=[]
        while len(has_weight)!=len(nodes):
            for node in nodes:
                if not node in has_weight:
                    in_edges=decay_graph.in_edges(node,'weight')
                    in_nodes=[n[0] for n in in_edges]
                    if set(in_nodes).issubset(has_weight) and not node in has_weight:
                        for edge in in_edges:
                            nodes[node]['weight']+=nodes[edge[0]]['weight']*edge[2]
                        has_weight.append(node)
                        print(f'{len(has_weight)} of {len(nodes)} nodes done',end="\r")
        return decay_graph

    def simulateDecay(self,scalar_mass):
        print('Building decay graph...')
        decay_graph,initial_states=self.build_decay_graph(scalar_mass)
        print(f'Generated decay graph with {decay_graph.number_of_nodes()} nodes and {decay_graph.number_of_edges()} edges.')
        print('Building weights...')
        weighted_graph=self.buildWeights(scalar_mass,decay_graph,initial_states)
        print('\n')
        print('Done')
        return weighted_graph

    def get_final_states(self,decay_graph):
        final_states= [s for s,d in decay_graph.out_degree() if d==0]
        attributes=nx.get_node_attributes(decay_graph,'weight')
        return {f:attributes[f] for f in final_states}

    def get_most_common_final_states(self,decay_graph):
        final_states=self.get_final_states(decay_graph)
        return {k: v for k, v in sorted(final_states.items(), key=lambda item: item[1],reverse=True)}

    def get_latex_id(self,list_of_ids):
        if not list_of_ids: return 'none'
        else:
            count=sorted(Counter(list_of_ids).items(),key=lambda x: np.abs(x[0]),reverse=True)
            vid=r'$'
            for x in count:
                if x[1]>1:
                    vid+=str(x[1])
                vid+=Particle.from_pdgid(x[0]).latex_name
            return vid+'$'

    def plot_from_initial_state(self,decay_graph,state):
        descendants=nx.descendants(decay_graph,state)
        subgraph = decay_graph.subgraph(descendants.union({state}))
        self.plot_from_init_final_state_helper(subgraph)

    def plot_from_final_state(self,decay_graph,state):
        ancestors=nx.ancestors(decay_graph,state)
        subgraph = decay_graph.subgraph(ancestors.union({state}))
        self.plot_from_init_final_state_helper(subgraph)

    def plot_from_init_final_state_helper(self,subgraph):
        import matplotlib.pyplot as plt 
        from matplotlib.patches import FancyArrowPatch

        round_val=4

        pos = nx.drawing.nx_agraph.graphviz_layout(subgraph, prog="dot", args="-Grankdir=LR")
        edge_labels = {key:round(val,round_val) for key,val in nx.get_edge_attributes(subgraph,'weight').items()}
        node_labels = {key:self.get_latex_id(key)+'\n'+str(round(val,round_val)) for key,val in nx.get_node_attributes(subgraph,'weight').items()}
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_edges(subgraph, pos, edgelist=subgraph.edges(), edge_color='gray',arrows=False)
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=10, label_pos=0.45)

        # Draw the node labels with multi-line text
        for node, (x, y) in pos.items():
            plt.text(x, y, node_labels[node], ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.2'))


        nx.draw_networkx_nodes(subgraph, pos, node_size=0)


        for (u, v) in subgraph.edges():
            # Calculate the midpoint of the edge
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
            edge_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            direction = np.arctan2(y2 - y1, x2 - x1)
            shift_distance = 0.1*edge_length  # Adjust this value to move the arrowhead further
            arrow_x = midpoint[0] + shift_distance * np.cos(direction)
            arrow_y = midpoint[1] + shift_distance * np.sin(direction)

            # Draw the arrow
            arrow = FancyArrowPatch(
                (arrow_x, arrow_y),
                (arrow_x + 0.001 * np.cos(direction), arrow_y + 0.001 * np.sin(direction)),
                arrowstyle='-|>', mutation_scale=20, color='gray')
                    # Calculate the direction of the arrow

            plt.gca().add_patch(arrow)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.axis('off')
        plt.show()

    def plot_hist(self,decay_graph,scalar_mass,n=10):
        fig, ax=plt.subplots()
        final_states=self.get_most_common_final_states(decay_graph)
        labels=list(final_states.keys())[:n]
        values=list(final_states.values())[:n]
        latex_ids=[self.get_latex_id(x) for x in labels]
        x_axis=range(n)
        ax.set_xticks(x_axis, [textwrap.fill(label, 10,break_long_words=False) for label in latex_ids], rotation = 20, fontsize=8, horizontalalignment="center")
        ax.bar(x_axis,values)
        #ax.text(x_axis[-2],hist[0], 'mh='+str(scalar_mass)+'MeV\n', horizontalalignment='center',verticalalignment='top')
        fig.tight_layout()
        plt.show()

SH=ScalarHadronizer()
decay_graph=SH.simulateDecay(2000)
SH.plot_hist(decay_graph,2000)
SH.plot_from_final_state(decay_graph,(-211,-211,111,111,211,211))
#SH.plot_from_initial_state(decay_graph,(-213,213))
#init_pairs=SH.make_initialMesonPairs(2000)
# %%
SH.get_final_states(decay_graph)
SH.plot_from_final_state(decay_graph,(-211,111,111,211))

# %%
SH.check_meson_combinations(Particle.from_pdgid(-213),Particle.from_pdgid(213),2000)
(-213,213) in SH.make_initialMesonPairs(2000).keys()
# %%
pi=Particle.findall('pi+')[0]
print(pi)
print(pi.pdgid)
print(pi.invert()==pi)
print(pi.invert().pdgid)
print(pi.pdgid.has_up)
print(pi.pdgid.has_down)
print(pi.pdgid.has_strange)
print(pi.quarks)
import particle.pdgid
#print(particle.pdgid.has_up(111))
# %%




# %%

def brs_consistency_check(hadronizer,decay_graph,scalar_mass):
    for node in decay_graph.nodes:
        decays=hadronizer.all_decays_of_multiparticle_state(node)
        br=0
        out_edges=decay_graph.out_edges(node,'weight')
        for edge in out_edges:
            br+=edge[2]
        for d in out_edges:
            assert d[1] in decays.keys(), f'Decay {d} is missing from outgoing edges for node {node}.\n Decays: {decays} \n outgoing edges: {out_edges}'
        if out_edges: assert br>0.99, f'Branching ratio of state {node} is only {br}, with edges {out_edges}'

def weight_consistency_check(hadronizer,decay_graph,scalar_mass):
    initial_states=hadronizer.make_initialMesonPairs(scalar_mass)
    for node in decay_graph.nodes:
        w=0
        edges=decay_graph.in_edges(node,'weight')
        for edge in edges:
            w+=decay_graph.nodes[edge[0]]['weight']*edge[2]
        if node in initial_states.keys(): w+=initial_states[node]
        weight=decay_graph.nodes[node]['weight']
        assert abs(weight-w)<1e-5,f'Wrong weight of node {node}: Is {weight}, should be {w}'
    assert not 0 in [v for u,v in decay_graph.nodes.data('weight')]

def total_brs_consistency_check(hadronizer,decay_graph):
    final_states=hadronizer.get_final_states(decay_graph)
    assert abs(np.sum(list(final_states.values()))-1)<1e-5

SH=ScalarHadronizer()
mass=4000
decay_graph=SH.simulateDecay(mass)
brs_consistency_check(SH,decay_graph,mass)
weight_consistency_check(SH,decay_graph,mass)
total_brs_consistency_check(SH,decay_graph)

# %%
%load_ext line_profiler
SH=ScalarHadronizer()
mass=3000
%lprun -f SH.buildWeights SH.simulateDecay(mass)

# %%



