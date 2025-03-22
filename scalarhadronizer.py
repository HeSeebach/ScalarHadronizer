from particle import Particle 
from particle.pdgid import has_up,has_down,has_strange,has_charm,has_bottom
import numpy as np 
import xml.etree.ElementTree as ET
import itertools
from math import prod
from bs4 import BeautifulSoup
from collections import Counter
import networkx as nx 
import textwrap
import logging
from decay_widths import gamma_gg,gamma_ss
from os import path,remove
#logger = logging.getLogger(__name__)
#logging.basicConfig(filename='log.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


class ScalarHadronizer:
    def __init__(self,scalar_mass,spin_suppression=1,up_weight=1,down_weight=1,strange_weight=1,charm_weight=0,bottom_weight=0,gamma_fac=1,suppression_mode='spin',path_to_decayXML='DECAY.XML'):

        if path.exists('log.log'):
            remove('log.log')

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('log.log', mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.info(f"ScalarHadronizer instance created with scalar_mass={scalar_mass}")

        self.all_decaysXML=self.read_decayXML(path_to_decayXML)
        self.set_stable=[111]

        self.decay_graph=None

        self.scalar_mass=scalar_mass
        self.spin_suppression=spin_suppression
        self.up_weight=up_weight
        self.down_weight=down_weight
        self.strange_weight=strange_weight
        self.charm_weight=charm_weight
        self.bottom_weight=bottom_weight
        self.gamma_fac=gamma_fac

        self.neutral_light_meson_quark_content=self.neutral_light_meson_mixing()

        self.suppression_mode=suppression_mode


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
                self.logger.info(f'Invalid particle {p["name"]} in DECAY.XML')
        return all_decaysXML

    def neutral_light_meson_mixing(self):
        #mixing angles taken from herwig as implemented in the HadronSelector.cc file
        eta_mix=-23*np.pi/180
        phi_mix=36*np.pi/180
        neutral_light_mesons=   {111: [0.5,0.5,0,0,0],                                                              #pi0 
                                 221: [0.5*np.cos(eta_mix)**2,0.5*np.cos(eta_mix)**2,np.sin(eta_mix)**2,0,0],       #eta
                                 331: [0.5*np.sin(eta_mix)**2,0.5*np.sin(eta_mix)**2,np.cos(eta_mix)**2,0,0],       #eta'
                                 113: [0.5,0.5,0,0,0],                                                              #rho0 
                                 333: [0.5*np.cos(eta_mix)**2,0.5*np.cos(eta_mix)**2,np.sin(eta_mix)**2,0,0],       #phi
                                 223: [0.5*np.sin(eta_mix)**2,0.5*np.sin(eta_mix)**2,np.cos(eta_mix)**2,0,0]}       #omega
        return neutral_light_mesons

    def set_parameters(self,spin_suppression=1,up_weight=1,down_weight=1,strange_weight=1,charm_weight=0,bottom_weight=0,gamma_fac=1):
        self.spin_suppression=spin_suppression
        self.up_weight=up_weight
        self.down_weight=down_weight
        self.strange_weight=strange_weight
        self.charm_weight=charm_weight
        self.bottom_weight=bottom_weight
        self.gamma_fac=gamma_fac

    def single_particle_decays(self,pdgid: int, logging=True):
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
                    try:
                        daughters=tuple([int(Particle.from_evtgen_name(x).pdgid) for x in channel['daughters'].split()])
                    except:
                        branching_ratios={}
                        d=channel['daughters']
                        if logging: self.logger.info(f'Invalid decay into {d} for particle {Particle.from_pdgid(pdgid)},{pdgid}')
                        break
                    branching_ratios[daughters]=branching_ratio
                    total_br+=branching_ratio
            branching_ratios={d:b/total_br for d,b in branching_ratios.items()}
        return branching_ratios
    
    def is_anti_particle(self,p1,p2):
        return int(p1.pdgid)==-int(p2.pdgid)

    def is_neutral(self,p):
        return p.anti_flag.name=='Same'

    def are_antiparticles(self,p1,p2):
        return int(p1.pdgid)==-int(p2.pdgid)

    def check_meson_combinations(self, m1, m2, m):
        mesons_to_exclude = [130, 310, 9000221, 9000321, -9000321, 9000311, -9000311]  # exclude KL and KS and some others

        # Check mass and exclusion list
        if m1.mass + m2.mass > m or m1.pdgid in mesons_to_exclude or m2.pdgid in mesons_to_exclude: return False

        # Check C conservation for particle - antiparticle
        if (not self.is_neutral(m1) or not self.is_neutral(m2)) and not self.are_antiparticles(m1, m2): return False

        # Check C and P for neutral mesons
        if self.is_neutral(m1) and self.is_neutral(m2):
                if m1.C!=m2.C: return False                                                                      #check charge conjugation eigenvalue
                if m1.J==0 and m2.J==0 and m1.P!=m2.P: return False
                if ((m1.J==1 and m2.J==0) or (m1.J==0 and m2.J==1)) and m1.P!=(-1)*m2.P: return False            #parity must be opposite for l=1
                if ((m1.J==2 and m2.J==0) or (m1.J==0 and m2.J==2)) and m1.P!=m2.P: return False                 #parity must be same for l=2
                if ((m1.J==3 and m2.J==0) or (m1.J==0 and m2.J==3)) and m1.P!=(-1)*m2.P: return False            #parity must be opposite for l=3

        # Check isospin conservation
        if m1.I != m2.I: return False
        return True

    def make_initialMesonPairs(self,mesons_below_threshold=None,exclude_below_threshold=True,up_weight=None,down_weight=None,strange_weight=None,charm_weight=None,bottom_weight=None,spin_suppression=None,gamma_fac=None):
        if up_weight is None: up_weight=self.up_weight
        if down_weight is None: down_weight=self.down_weight
        if strange_weight is None: strange_weight=self.strange_weight
        if charm_weight is None: charm_weight=self.charm_weight
        if bottom_weight is None: bottom_weight=self.bottom_weight
        if spin_suppression is None: spin_suppression=self.spin_suppression
        if gamma_fac is None: gamma_fac=self.gamma_fac

        if mesons_below_threshold is None: mesons_below_threshold=Particle.findall(lambda p: p.mass<self.scalar_mass and p.pdgid.is_meson==True)
        if not mesons_below_threshold: print('No possible decay products. Maybe mass is too small? (should be in MeV)')
        meson_pairs={}
        total_gluon_initial_weight=0
        total_strange_initial_weight=0
        for i,m1 in enumerate(mesons_below_threshold):
            for m2 in mesons_below_threshold[i:]:
                if self.check_meson_combinations(m1,m2,self.scalar_mass):
                    g_weight,s_weight=self.initialWeight(m1,m2,up_weight,down_weight,strange_weight,charm_weight,bottom_weight,spin_suppression,gamma_fac)
                    if m1.pdgid<m2.pdgid: meson_pairs[(int(m1.pdgid),int(m2.pdgid))]=[g_weight,s_weight]
                    else: meson_pairs[(int(m2.pdgid),int(m1.pdgid))]=[g_weight,s_weight]
                    total_gluon_initial_weight+=g_weight
                    total_strange_initial_weight+=s_weight
        
        gg_BR=gamma_gg(self.scalar_mass*1e-3)/(gamma_gg(self.scalar_mass*1e-3)+gamma_ss(self.scalar_mass*1e-3))
        ss_BR=gamma_ss(self.scalar_mass*1e-3)/(gamma_gg(self.scalar_mass*1e-3)+gamma_ss(self.scalar_mass*1e-3))
        meson_pairs={k: (gg_BR*v[0]/total_gluon_initial_weight + ss_BR *v[1]/total_strange_initial_weight) for k,v in meson_pairs.items()}
        if exclude_below_threshold:
            meson_pairs={k:v for k,v in meson_pairs.items() if v>1e-3}
            new_total_weight=np.sum(list(meson_pairs.values()))
            meson_pairs={k:v/new_total_weight for k,v in meson_pairs.items()}
        return meson_pairs

    def initialWeight(self,m1,m2,up_weight,down_weight,strange_weight,charm_weight,bottom_weight,spin_suppression,gamma_fac,logging=True):
        if self.neutral_meson_quark_content(int(m1.pdgid))[2]>0 or self.neutral_meson_quark_content(int(m2.pdgid))[2]>0: ss_channel_quark_weight=self.ss_channel_quark_weight(m1,m2,up_weight,down_weight,strange_weight,charm_weight,bottom_weight,spin_suppression,gamma_fac)
        else: ss_channel_quark_weight=0
        gluon_channel_quark_weight=self.gg_channel_quark_weight(m1,m2,up_weight,down_weight,strange_weight,charm_weight,bottom_weight,spin_suppression,gamma_fac)
        p,spin_fac,isospin_fac,sym_fac=self.rest_of_initial_weight(m1,m2,up_weight,down_weight,strange_weight,charm_weight,bottom_weight,spin_suppression,gamma_fac)
        gluon_initial_weight=p*gluon_channel_quark_weight*spin_fac*isospin_fac*sym_fac
        strange_initial_weight=p*ss_channel_quark_weight*spin_fac*isospin_fac*sym_fac

        if logging:
            self.logger.info(f'{m1.name} {m2.name}:')
            self.logger.info(f'\t Gluon channel weight: {gluon_initial_weight}. p_restframe: {p}, quark_weight: {gluon_channel_quark_weight}, spin_factor: {spin_fac}, isospin_factor: {isospin_fac}')
            self.logger.info(f'\t Strange channel weight: {strange_initial_weight}. p_restframe: {p}, quark_weight: {ss_channel_quark_weight}, spin_factor: {spin_fac}, isospin_factor: {isospin_fac}')
        return gluon_initial_weight,strange_initial_weight

    def ss_channel_quark_weight(self,m1,m2,up_weight,down_weight,strange_weight,charm_weight,bottom_weight,spin_suppression,gamma_fac):
        if self.is_neutral(m1):
            m1_quark_content=self.neutral_meson_quark_content(m1.pdgid)
            m2_quark_content=self.neutral_meson_quark_content(m2.pdgid)
            quark_weight=m1_quark_content[2]*m2_quark_content[2]*strange_weight
        else:
            quark_weight=1
            if m1.pdgid.has_up: quark_weight*=up_weight
            if m1.pdgid.has_down: quark_weight*=down_weight
            if m1.pdgid.has_charm: quark_weight*=charm_weight
            if m1.pdgid.has_bottom: quark_weight*=bottom_weight
        return quark_weight

    def gg_channel_quark_weight(self,m1,m2,up_weight,down_weight,strange_weight,charm_weight,bottom_weight,spin_suppression,gamma_fac):
        if self.is_neutral(m1):
            quark_weight=0
            for a,b,w in zip(self.neutral_meson_quark_content(m1.pdgid),self.neutral_meson_quark_content(m2.pdgid),[up_weight,down_weight,strange_weight,charm_weight,bottom_weight]):
                quark_weight+=a*b*w**2
        else:
            quark_weight=1
            if m1.pdgid.has_up: quark_weight*=up_weight
            if m1.pdgid.has_down: quark_weight*=down_weight
            if m1.pdgid.has_strange: quark_weight*=strange_weight
            if m1.pdgid.has_charm: quark_weight*=charm_weight
            if m1.pdgid.has_bottom: quark_weight*=bottom_weight
        return quark_weight
        
    def rest_of_initial_weight(self,m1,m2,up_weight,down_weight,strange_weight,charm_weight,bottom_weight,spin_suppression,gamma_fac):
        p_restframe=np.sqrt((self.scalar_mass**2-(m1.mass+m2.mass)**2)*(self.scalar_mass**2-(m1.mass-m2.mass)**2))/2/self.scalar_mass
        if np.isnan(p_restframe): print(f'Meson pair {m1.name},{m2.name} too heavy')

        #spin multiplicity
        if self.suppression_mode=='spin':
            spin_factor=(2*m1.J+1)*(2*m2.J+1)
            if not (m1.J==0 and m2.J==0): spin_factor*=spin_suppression
        if self.suppression_mode=='OAM':
            if m1.J==m2.J: spin_factor=(2*m1.J+1)+spin_suppression*((2*m1.J+1)*(2*m2.J+1)-(2*m1.J+1))
            else: spin_factor=(2*m1.J+1)*(2*m2.J+1)*spin_suppression

        #isospin weight
        if m1.I==0.5 and m2.I==0.5: isospin_factor=0.5
        elif m1.I==1 and m2.I==1: isospin_factor=0.33
        else: isospin_factor=1

        #symmetry factor
        if m1==m2: symmetry_factor=1 #only to meson antimeson pairs or also for two different neutral mesons?
        else: symmetry_factor=2

        return p_restframe,spin_factor,isospin_factor,symmetry_factor

    def neutral_meson_quark_content(self,pdgid):
        if pdgid in self.neutral_light_meson_quark_content:
            return self.neutral_light_meson_quark_content[pdgid]
        else:
            quark_content=[0,0,0,0,0]
            if has_up(pdgid): quark_content[0]=1
            if has_down(pdgid): quark_content[1]=1
            if has_strange(pdgid): quark_content[2]=1
            if has_charm(pdgid): quark_content[3]=1
            if has_bottom(pdgid): quark_content[4]=1
            return quark_content

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

    def build_decay_graph(self,decay_graph=None,initial_states=None,exclude_below_threshold=True):
        if decay_graph is None:
            decay_graph=nx.DiGraph()
            initial_states=self.make_initialMesonPairs(exclude_below_threshold=exclude_below_threshold)
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

        if not all_decays_finished: decay_graph,initial_states=self.build_decay_graph(decay_graph,initial_states)
        return decay_graph,initial_states

    def make_weight(self,decay_graph,state,weight=0):
        in_edges=decay_graph.in_edges(state,'weight')
        if in_edges:
            for edge in in_edges:
                if decay_graph.nodes[edge[0]]['weight']==0:
                    weight+=self.make_weight(decay_graph,edge[0])*edge[2]
                else:
                    weight+=decay_graph.nodes[edge[0]]['weight']*edge[2]
        else:
            weight=decay_graph.nodes[state]['weight']
        if decay_graph.nodes[state]['weight']==0: decay_graph.nodes[state]['weight']=weight
        return weight

    def buildWeights(self,decay_graph,initial_states=None,final_states=None):
        if initial_states is None: initial_states=self.make_initialMesonPairs()
        if final_states is None: final_states= [s for s,d in decay_graph.out_degree() if d==0]
        nodes=decay_graph.nodes
        nx.set_node_attributes(decay_graph,0,name='weight')
        for s,w in initial_states.items():
            nodes[s]['weight']=w
        for state in final_states:
            w=self.make_weight(decay_graph,state)
        return decay_graph 

    def buildWeights2(self,decay_graph,initial_states):
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

    def simulateDecay(self):
        #print('Building decay graph...')
        decay_graph,initial_states=self.build_decay_graph()
        #print(f'Generated decay graph with {decay_graph.number_of_nodes()} nodes and {decay_graph.number_of_edges()} edges.')
        #print('Building weights...')
        #weighted_graph=self.buildWeights(decay_graph,initial_states)
        weighted_graph=self.buildWeights(decay_graph)
        #print('\n')
        #print('Done')
        self.decay_graph=weighted_graph
        return weighted_graph

    def initialize_meson_list_for_parameter_fits(self):

        mesons_below_threshold=Particle.findall(lambda p: p.mass<self.scalar_mass and p.pdgid.is_meson==True)
        meson_pairs=[]
        for i,m1 in enumerate(mesons_below_threshold):
            for m2 in mesons_below_threshold[i:]:
                if self.check_meson_combinations(m1,m2,self.scalar_mass):
                    if m1.pdgid<m2.pdgid: meson_pairs.append((m1,m2))
                    else: meson_pairs.append((m2,m1))
        return meson_pairs

    def parameter_fit_func(self,ws,wv,gamma_fac,list_of_meson_pairs):
        meson_pairs={}
        total_gluon_initial_weight=0
        total_strange_initial_weight=0
        for m1,m2 in list_of_meson_pairs:
            g_weight,s_weight=self.initialWeight(m1,m2,self.up_weight,self.down_weight,ws,0,0,wv,gamma_fac)
            if m1.pdgid<m2.pdgid: meson_pairs[(int(m1.pdgid),int(m2.pdgid))]=[g_weight,s_weight]
            else: meson_pairs[(int(m2.pdgid),int(m1.pdgid))]=[g_weight,s_weight]
            total_gluon_initial_weight+=g_weight
            total_strange_initial_weight+=s_weight
        
        gg_BR=gamma_gg(self.scalar_mass*1e-3)/(gamma_gg(self.scalar_mass*1e-3)+gamma_ss(self.scalar_mass*1e-3))
        ss_BR=gamma_ss(self.scalar_mass*1e-3)/(gamma_gg(self.scalar_mass*1e-3)+gamma_ss(self.scalar_mass*1e-3))
        meson_pairs={k: (gg_BR*v[0]/total_gluon_initial_weight + ss_BR *v[1]/total_strange_initial_weight) for k,v in meson_pairs.items()}
        br_pi=meson_pairs[(-211,211)]+meson_pairs[(111,111)]
        br_K=meson_pairs[(-321,321)]+meson_pairs[(-311,311)]
        gamma_pi=self.get_decay_width(br_pi,gamma_fac)
        gamma_K=self.get_decay_width(br_K,gamma_fac)
        return br_pi,br_K,gamma_pi,gamma_K 

    def get_decay_width(self,branching_ratio,gamma_fac=None):
        if gamma_fac is None: gamma_fac=self.gamma_fac
        return branching_ratio*gamma_fac*(gamma_gg(self.scalar_mass*1e-3)+gamma_ss(self.scalar_mass*1e-3))

    def get_final_states(self,decay_graph=None):
        if decay_graph is None: decay_graph=self.decay_graph
        final_states= [s for s,d in decay_graph.out_degree() if d==0]
        attributes=nx.get_node_attributes(decay_graph,'weight')
        return {f:attributes[f] for f in final_states}

    def get_most_common_final_states(self,decay_graph=None):
        if decay_graph is None: decay_graph=self.decay_graph
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

    def print_initial_states(self,more_info=False,exclude_below_threshold=True):
        initial_states=self.make_initialMesonPairs(exclude_below_threshold=exclude_below_threshold)
        print('The following initial states were generated:')
        sorted_states = dict(sorted(initial_states.items(), key=lambda item: item[1], reverse=True))
        for key, value in sorted_states.items():
            p1=Particle.from_pdgid(key[0])
            p2=Particle.from_pdgid(key[1])
            print(f'{p1.name}[{int(p1.pdgid)}] {p2.name}[{int(p2.pdgid)}]: {value}')
            if more_info:
                    print(f'\t J: {p1.J} {p2.J}, I: {p1.I} {p2.I}, C: {p1.C} {p2.C}, P: {p1.P} {p2.P}')
    
    def print_final_states(self,n,decay_graph=None):
        if decay_graph is None: decay_graph=self.decay_graph
        final_states=self.get_most_common_final_states(decay_graph)
        print('The following final states were generated:')
        for i,key in enumerate(final_states):
            if i<n:
                print(key, end=' ')
                for p in key:
                    print(Particle.from_pdgid(p).name,end=' ')
                print(': ',final_states[key])

    def get_ancestors_of_state(self,state,decay_graph=None):
        if decay_graph is None: decay_graph=self.decay_graph
        ancestors=nx.ancestors(decay_graph,state)
        attributes=nx.get_node_attributes(decay_graph,'weight')
        return {ancestor: nx.shortest_path_length(decay_graph, ancestor, state) for ancestor in ancestors}

    def print_ancestors_of_state(self, state, decay_graph=None):
        if decay_graph is None: decay_graph = self.decay_graph
        ancestors = self.get_ancestors_of_state(state, decay_graph)
        sorted_ancestors = sorted(ancestors.items(), key=lambda item: item[1])
        current_distance = None
        initial_states=self.make_initialMesonPairs()
        for ancestor, distance in sorted_ancestors:
            if distance != current_distance:
                print(f"---Distance {distance}--------------------------------")
                current_distance = distance
            indent = ' ' * (distance * 4)
            print(indent,end='')
            for a in ancestor:
                print(Particle.from_pdgid(a).name,end=' ')
            if ancestor in initial_states: print(' [i]',end='')
            print('')

    def plot_from_initial_state(self,state,decay_graph=None,path=None):
        if decay_graph is None: decay_graph=self.decay_graph
        descendants=nx.descendants(decay_graph,state)
        subgraph = decay_graph.subgraph(descendants.union({state}))
        self.plot_from_init_final_state_helper(subgraph,path)

    def plot_from_final_state(self,state,decay_graph=None,path=None):
        if decay_graph is None: decay_graph=self.decay_graph
        ancestors=nx.ancestors(decay_graph,state)
        subgraph = decay_graph.subgraph(ancestors.union({state}))
        self.plot_from_init_final_state_helper(subgraph,path)

    def plot_from_init_final_state_helper(self,subgraph,path=None):
        import matplotlib.pyplot as plt 
        from matplotlib.patches import FancyArrowPatch

        round_val=4

        # Identify initial nodes (nodes with no incoming edges)
        #initial_nodes = [node for node in subgraph.nodes() if subgraph.in_degree(node) == 0]
        final_nodes = [node for node in subgraph.nodes() if subgraph.out_degree(node) == 0]

        # Calculate the distance of each node from the initial nodes
        distances = {node: float('inf') for node in subgraph.nodes()}
        for final_node in final_nodes:
            distances[final_node] = 0
            for node in nx.ancestors(subgraph, final_node):
                distances[node] = min(distances[node], nx.shortest_path_length(subgraph, node, final_node))
        for node, distance in distances.items():
            subgraph.nodes[node]['rank'] = str(distance)

        # Use graphviz_layout with dot and rankdir options
        pos = nx.drawing.nx_agraph.graphviz_layout(subgraph, prog="dot", args="-Grankdir=LR -Gnodesep=0.5 -Granksep=1.0")
        #pos = nx.drawing.nx_agraph.graphviz_layout(subgraph, prog="dot", args="-Grankdir=LR")
        edge_labels = {key:round(val,round_val) for key,val in nx.get_edge_attributes(subgraph,'weight').items()}
        node_labels = {key:self.get_latex_id(key)+'\n'+str(round(val,round_val)) for key,val in nx.get_node_attributes(subgraph,'weight').items()}
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_edges(subgraph, pos, edgelist=subgraph.edges(), edge_color='gray',arrows=False)
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=10, label_pos=0.45)

        # Draw the node labels with multi-line text
        for node, (x, y) in pos.items():
            plt.text(x, y, node_labels[node], ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.2'))


        nx.draw_networkx_nodes(subgraph, pos,node_size=2000, node_color='white')


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
        if not path is None: plt.savefig(path)
        plt.show()

    def plot_final_state_hist(self,decay_graph=None,n=10,show=True):
        if decay_graph is None: decay_graph=self.decay_graph
        import matplotlib.pyplot as plt 
        fig, ax=plt.subplots()
        final_states=self.get_most_common_final_states(decay_graph)
        labels=list(final_states.keys())[:n]
        values=list(final_states.values())[:n]
        latex_ids=[self.get_latex_id(x) for x in labels]
        x_axis=range(n)
        ax.set_xticks(x_axis, [textwrap.fill(label, 10,break_long_words=False) for label in latex_ids], rotation = 20, fontsize=8, horizontalalignment="center")
        ax.bar(x_axis,values)
        ax.text(x_axis[-2],values[0], r'$m_\phi=$'+str(self.scalar_mass*1e-3)+'GeV\n', horizontalalignment='center',verticalalignment='top')
        fig.tight_layout()
        #plt.title(f'Branching ratios of {n} most common final states')
        if show: plt.show()
        return fig,ax

    def plot_initial_state_hist(self,n=10):
        import matplotlib.pyplot as plt 
        initial_states=self.make_initialMesonPairs()
        sorted_states = dict(sorted(initial_states.items(), key=lambda item: item[1], reverse=True))
        if n>len(initial_states): n=len(initial_states)
        fig, ax=plt.subplots()
        labels=list(sorted_states.keys())[:n]
        values=list(sorted_states.values())[:n]
        latex_ids=[self.get_latex_id(x) for x in labels]
        x_axis=range(n)
        ax.set_xticks(x_axis, [textwrap.fill(label, 10,break_long_words=False) for label in latex_ids], rotation = 20, fontsize=8, horizontalalignment="center")
        ax.bar(x_axis,values)
        ax.text(x_axis[-2],values[0], 'm='+str(self.scalar_mass*1e-3)+'GeV\n', horizontalalignment='center',verticalalignment='top')
        fig.tight_layout()
        plt.title(f'Branching ratios of {n} most common initial states')
        plt.show()

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
