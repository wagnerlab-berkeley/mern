from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
import tempfile
import networkx as nx
from anndata import AnnData
import pickle
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

from Bio.KEGG import REST
from Bio.KEGG.KGML import KGML_parser
import Bio.KEGG
import os
import mygene


from . import config

def process_kegg_link(linked, from_item, to_item):
    """
    KEGG API helper function
    """
    linked = linked.read().split('\n')[:-1]
    firsts = [first.split('\t')[0] for first in linked]
    seconds = [second.split('\t')[1] for second in linked]
    final = pd.DataFrame({from_item:firsts, to_item:seconds})
    return final

class MetabolicDataset(ABC):
    """
    Abstract class for metabolic data sets that will provide metabolic topology

    Parameters
    ----------
    species
        which species to use for dataset
    capitalize
        if true, capitalize all genes
    """

    def __init__(self, species : str, data_dir : str, capitalize : bool = False) -> None:
        self.species = species
        self.data_dir = data_dir
        self.capitalize_genes = capitalize

    @abstractmethod
    def metabolic_topology(
        self,
        rna: AnnData,
        self_loops: bool = True,
    ) -> nx.DiGraph:
        """
        Composes directed rxn-rxn graph based on metabolic topology.
        
        Parameters
        ----------
        rna
            RNA dataset
        self_loops
            whether or not to include self loop
        
        Returns
        --------
        graph
            metabolic topology graph
        """
        raise NotImplementedError
    
    @abstractmethod
    def add_module_info(
        self,
        rna: AnnData,
    ):
        """
        Add metabolic module information for each gene in data set
        
        Parameters
        -----------
        rna
            anndata object
        """
        raise NotImplementedError
    
    @abstractmethod
    def add_rxn_module_info(
        self,
        rna: AnnData,
        graph: nx.DiGraph,
    ):
        """
        Add module information on graph nodes to uns
        
        Parameters
        ---------
        rna
            anndata object
        graph
            metabolic topology graph (rxn)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_rxn_genes(
        self,
        rna: AnnData,
        subset_genes: bool = True,
    ):
        """
        Gets dictionary mapping rxns to genes

        Parameters
        ----------
        rna
            RNA dataset
        subset_genes
            subset to only genes in data set
    
        Returns
        --------
        rxn_genes
            Mapping from rxns to genes
        """
        raise NotImplementedError
    
    @abstractmethod
    def flatten_rxns(self):
        """
        Args:
            graphs (dict): dictionary containing edge lists of graphs for each module
        Returns:
            all_rxns (list): list of all included rxns
        """
        raise NotImplementedError
    
    @abstractmethod
    def calculate_stoichiometric_mat(
        self
    ) -> pd.DataFrame:
        """
        Calculates stoichiometric matrix

        Parameters
        ----------
        species
            species to use

        Returns
        -------
        stoichiometric matrix as df
        """
        raise NotImplementedError
    
    @abstractmethod
    def calculate_metabolites(
        self,
        rna: AnnData,
        absolute: bool = False,
    ):
        """
        Calculates relative metabolite activations based on rxn activations. Modifies anndata object

        Parameters
        ---------
        rna
            anndata object with module info
        
        Returns
        --------
        met_df
            df with cells by metabolites
        """
        raise NotImplementedError

class KeggKGMLMetabolicDataset(MetabolicDataset):
    """
    Metabolic dataset based on KEGG KGML files. Provides functionality for parsing and analyzing
    metabolic reactions and pathways from KEGG KGML files.
    """

    OXPHOS_RXNS = {
        'rn:R11945': {
            'substrates': ['cpd:C00399', 'cpd:C00004'],
            'products': ['cpd:C00390', 'cpd:C00003'],
        },
        'rn:R13223': {
            'substrates': ['cpd:C00399', 'cpd:C00042'],
            'products': ['cpd:C00390', 'cpd:C00122'],
        },
        'rn:R13224': {
            'substrates': ['cpd:C15603', 'cpd:C00125'],
            'products': ['cpd:C15602', 'cpd:C00126'],
        },
        'rn:R02161': {
            'substrates': ['cpd:C00390', 'cpd:C00125'],
            'products': ['cpd:C00399', 'cpd:C00126'],
        },
        'rn:R00081': {
            'substrates': ['cpd:C00126'],
            'products': ['cpd:C00125'],
        },
    }

    def __init__(
        self,
        species: str,
        capitalize: bool = False,
        add_oxphos: bool = True,
        keep_isolates: bool = False,
    ):
        """
        Initialize KEGG KGML metabolic dataset

        Parameters
        ----------
        species
            Species name (e.g. 'human', 'mouse')
        capitalize
            Whether to capitalize gene names
        """
        super().__init__(species, config.KEGG_CACHE_DIR, capitalize)

        self.package_data_dir = config.KEGG_DIR
        self.add_oxphos = add_oxphos
        self.keep_isolates = keep_isolates
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        if self.species == 'human':
            self.kegg_species = 'hsa'
            graph_name = 'human_metabolic_graph.pkl'
        elif self.species == 'mouse':
            self.kegg_species = 'mmu'
            graph_name = 'mouse_metabolic_graph.pkl'
        else:
            raise ValueError('Not a valid species for KEGG')

        with open(Path(self.package_data_dir) / graph_name, 'rb') as f:
            self._bundled_graph = pickle.load(f)

        self.kgml = self._load_or_download_kgml()
        if add_oxphos:
            self.kgml = self.add_oxphos_rxns(self.kgml)
        self.rxns = list(self.kgml.reactions)
        self.rxn_info = self.get_kgml_rxn_info()
        self.rxn_genes = self.get_rxn_genes_all()
        self.all_compounds = self.get_all_compounds()
        self.compound_info = self.get_compound_info()
        self.pathway_kgmls = {}

    def _download_notice(self, item: str, cache_location: str = None):
        if cache_location is None:
            cache_location = self.data_dir
        print(
            f'Initiating one-time download from KEGG for {self.species}: {item}. '
            'This uses KEGG access provided for academic use by academic users; '
            "you are responsible for complying with KEGG's terms. "
            f'the result will be cached in {cache_location} and reused.'
        )

    @staticmethod
    def _atomic_write_text(path: Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False
            ) as f:
                temp_path = Path(f.name)
                f.write(text)
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _atomic_pickle_dump(value, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='wb', dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False
            ) as f:
                temp_path = Path(f.name)
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def _load_or_download_kgml(self):
        path = Path(self.data_dir) / f'{self.kegg_species}01100.kgml'
        if path.exists():
            with path.open() as f:
                return KGML_parser.read(f)

        self._download_notice(f'{self.kegg_species}01100 KGML')
        kgml_text = REST.kegg_get(f'{self.kegg_species}01100', 'kgml').read()

        # Parse before caching so an invalid or incomplete response is never installed.
        kgml = KGML_parser.read(StringIO(kgml_text))
        self._atomic_write_text(path, kgml_text)
        return kgml
    
    def add_oxphos_rxns(self, kgml):
        """
        Add oxphos reactions to kgml
        """
        oxphos_rxns = self.OXPHOS_RXNS
        # Find compounds by name/ID
        def find_compound_id(pathway, compound_name):
            for entry in pathway.entries.values():
                if entry.type == "compound" and entry.name == compound_name:
                    return entry.id
            return None

        # First, add all missing compounds to avoid ID clashes
        existing_ids = set(kgml.entries.keys())
        next_compound_id = max(existing_ids) + 1

        # Collect all unique compounds from oxphos reactions
        all_compounds = set()
        for rxn_data in oxphos_rxns.values():
            all_compounds.update(rxn_data['substrates'])
            all_compounds.update(rxn_data['products'])

        # Add missing compounds first
        compound_id_map = {}
        for compound in all_compounds:
            existing_id = find_compound_id(kgml, compound)
            if existing_id:
                compound_id_map[compound] = existing_id
            else:
                # Add missing compound to kgml
                compound_entry = Bio.KEGG.KGML.KGML_pathway.Entry()
                compound_entry.id = next_compound_id
                compound_entry.name = compound
                compound_entry.type = "compound"
                kgml.add_entry(compound_entry)
                compound_id_map[compound] = next_compound_id
                next_compound_id += 1

        # Now add all oxphos reactions
        existing_ids = set(kgml.entries.keys())
        next_reaction_id = max(existing_ids) + 1

        for rxn_id, rxn_data in oxphos_rxns.items():
            # Create new reaction
            reaction = Bio.KEGG.KGML.KGML_pathway.Reaction()
            
            reaction.id = next_reaction_id
            reaction.name = rxn_id  
            
            # Add substrates using pre-mapped compound IDs
            for substrate in rxn_data['substrates']:
                reaction.add_substrate(compound_id_map[substrate])
            
            # Add products using pre-mapped compound IDs
            for product in rxn_data['products']:
                reaction.add_product(compound_id_map[product])
            
            # Create corresponding entry for the reaction
            reaction_entry = Bio.KEGG.KGML.KGML_pathway.Entry()
            reaction_entry.id = reaction.id  # Same ID as reaction
            reaction_entry.name = reaction.name  # Same name as reaction
            reaction_entry.type = "reaction"  # Important: type is "reaction"
            
            # Add reaction to pathway
            kgml.add_entry(reaction_entry)
            kgml.add_reaction(reaction)
            
            next_reaction_id += 1

        return kgml
        
    def get_all_compounds(self):
        all_compounds = []  
        for rxn in self.rxns:
            for comp in rxn.substrates:
                all_compounds.append(comp.name)
            for comp in rxn.products:
                all_compounds.append(comp.name)
        all_compounds = list(set(all_compounds))
        return all_compounds
    
    def get_compound_info(self):
        path = Path(self.data_dir) / f'{self.kegg_species}_kgml_comp_info.pkl'
        if not path.exists():
            self._download_notice(f'{self.kegg_species} compound metadata')

            comp_info = {}
            for i in tqdm(range(0, len(self.all_compounds), 10)):
                comps = self.all_compounds[i:i+10]
                joined_comps = '+'.join(comps)
                joined_comps = joined_comps.replace(' ','+')
                comps = REST.kegg_get(joined_comps).read()    
                for comp in comps.split('///'):
                    info = self.parse_compound(comp)
                    if not info:
                        continue
                    else:
                        comp_info[info['id']] = info

            self._atomic_pickle_dump(comp_info, path)

        else:
            with path.open('rb') as f:
                comp_info = pickle.load(f)
        
        return comp_info
    
    def parse_compound(self, comp_str: str):
        info = {}
        current_field = None
        
        for line in comp_str.split('\n'):
            if not line.strip():
                continue
            if line.startswith(' '):  # Continuation of previous field
                if current_field in ['PATHWAY', 'MODULE']:
                    # For pathway and module, parse ID and name from continuation lines
                    if current_field in ['PATHWAY', 'MODULE']:
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2:
                            info[current_field].append((parts[0], parts[1]))
                        else:
                            info[current_field].append((parts[0], ''))
                    else:
                        info[current_field].append(line.strip())
            else:  # New field
                parts = line.split(None, 1)  # Split on first whitespace
                if len(parts) == 2:
                    field, value = parts
                    current_field = field
                        
                    if field == 'ENTRY':
                        info['id'] = value.split()[0]
                    elif field == 'NAME':
                        info['name'] = value
                    elif field == 'PATHWAY':
                        # Store both ID and name
                        pathway_parts = value.split(None, 1)
                        if len(pathway_parts) == 2:
                            info[field] = [(pathway_parts[0], pathway_parts[1])]
                        else:
                            info[field] = [(value, '')]
                    elif field == 'MODULE':
                        # Store both ID and name
                        module_parts = value.split(None, 1)
                        if len(module_parts) == 2:
                            info[field] = [(module_parts[0], module_parts[1])]
                        else:
                            info[field] = [(value, '')]
                    
        return info

    def get_rxn_genes_all(
        self,
        rebuild_from_kegg: bool = False,
    ):
        """
        Gets genes associated with reactions

        Parameters
        ----------
        rebuild_from_kegg
            download the current KEGG reaction-enzyme and enzyme-gene links
            instead of using the bundled reaction-gene mapping

        Returns
        -------
        rxn_genes
            dictionary of rxns to genes
        """
        path = Path(self.package_data_dir) / f'{self.kegg_species}_kgml_rxn_genes.pkl'
        if rebuild_from_kegg:
            self._download_notice(f'{self.kegg_species} reaction-gene links')

            reaction_enzyme = process_kegg_link(REST.kegg_link('enzyme', 'reaction'), 'Reaction', 'Enzyme')
            enzyme_gene = process_kegg_link(REST.kegg_link(self.kegg_species, 'enzyme'), 'Enzyme', 'Gene')

            rxn_gene = {}
            for rxn in self.rxns:
                rxn_genes = []

                try:
                    rxn_gene[rxn]
                except KeyError:
                    # get enzymes to genes
                    if len(rxn.name.split(' ')) > 1:
                        enzymes = pd.Series([])
                        for r in rxn.name.split(' '):
                            enzymes = pd.concat([enzymes, reaction_enzyme[reaction_enzyme['Reaction']==r]['Enzyme']])
                    else:
                        enzymes = reaction_enzyme[reaction_enzyme['Reaction']==rxn.name]['Enzyme']

                    for enzyme in enzymes:
                        #print(enzyme)
                        genes = enzyme_gene[enzyme_gene['Enzyme']==enzyme]['Gene']
                        for gene in genes:
                            #print(gene)
                            rxn_genes.append(gene)

                    rxn_gene[rxn] = rxn_genes

            all_genes = []
            for rxn in rxn_gene.keys():
                all_genes.extend(rxn_gene[rxn])

            all_genes = list(set(all_genes))
            all_genes = [gene.split(':')[1] for gene in all_genes]
            mg = mygene.MyGeneInfo()
            gene_res = mg.querymany(all_genes, species=self.species)
            assert len([gene for gene in gene_res if 'notfound' in gene]) == 0, 'Some genes not found'

            gene_res_dict = {item['query']: item for item in gene_res if 'query' in item}

            # get gene symbols
            rxn_gene_symbols = {}
            for rxn in tqdm(rxn_gene.keys()):
                genes = rxn_gene[rxn]
                #print(genes)
                gene_symbols = []
                for gene in genes:
                    gene_symbols.append(gene_res_dict[gene.split(':')[1]]['symbol'])

                rxn_gene_symbols[rxn.name] = gene_symbols

            cache_path = Path(self.data_dir) / f'{self.kegg_species}_kgml_rxn_genes.pkl'
            self._atomic_pickle_dump(rxn_gene_symbols, cache_path)
        else:
            if not path.exists():
                raise FileNotFoundError(f'Bundled reaction-gene mapping not found: {path}')

            with path.open('rb') as f:
                rxn_gene_symbols = pickle.load(f)

        if self.capitalize_genes:
            for rxn in rxn_gene_symbols.keys():
                temp_genes = rxn_gene_symbols[rxn]
                
                rxn_gene_symbols[rxn] = [gene.upper() for gene in temp_genes]
                
        return rxn_gene_symbols

    def get_kgml_rxn_info(self):
        """
        Gets KGML reaction info
        """
        path = Path(self.data_dir) / f'{self.kegg_species}_kgml_rxn_info.pkl'
        if not path.exists():
            self._download_notice(f'{self.kegg_species} reaction metadata')

            # flatten rxns
            flat_rxns = []
            for rxn in self.rxns:
                flat_rxns += rxn.name.split(' ')
            for node in self._bundled_graph.nodes:
                flat_rxns += node.split(' ')
            flat_rxns = list(dict.fromkeys(flat_rxns))

            rxn_info = {}
            for i in tqdm(range(0, len(flat_rxns), 10)):
                rxns = flat_rxns[i:i+10]
                joined_rxns = '+'.join(rxns)
                joined_rxns = joined_rxns.replace(' ','+')
                rxns = REST.kegg_get(joined_rxns).read()    
                for rxn in rxns.split('///'):
                    info = self.parse_rxn(rxn)
                    if not info:
                        continue
                    else:
                        rxn_info[info['id']] = info
            self._atomic_pickle_dump(rxn_info, path)

        else:
            with path.open('rb') as f:
                rxn_info = pickle.load(f)
        
        return rxn_info

    def parse_rxn(self, rxn_str: str) -> dict:
        """
        Parses KEGG reaction information string into a dictionary

        Parameters
        ----------
        rxn_str : str
            KEGG reaction information string

        Returns
        -------
        dict
            Dictionary containing parsed reaction information with fields:
            - id: reaction ID
            - name: reaction name
            - definition: reaction definition
            - equation: reaction equation with compound IDs
            - compounds: list of compound IDs
            - enzymes: list of EC numbers
            - pathways: list of pathway IDs
            - modules: list of module IDs
            - orthology: list of KO IDs
            - comments: list of comments
        """
        info = {}
        current_field = None
        
        for line in rxn_str.split('\n'):
            if not line.strip():
                continue
                
            if line.startswith(' '):  # Continuation of previous field
                if current_field in ['COMMENT', 'PATHWAY', 'MODULE', 'ENZYME', 'ORTHOLOGY']:
                    # For pathway and module, parse ID and name from continuation lines
                    if current_field in ['PATHWAY', 'MODULE']:
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2:
                            info[current_field].append((parts[0], parts[1]))
                        else:
                            info[current_field].append((parts[0], ''))
                    else:
                        info[current_field].append(line.strip())
            else:  # New field
                parts = line.split(None, 1)  # Split on first whitespace
                if len(parts) == 2:
                    field, value = parts
                    current_field = field
                    
                    if field == 'ENTRY':
                        info['id'] = value.split()[0]
                    elif field == 'NAME':
                        info['name'] = value
                    elif field == 'DEFINITION':
                        info['definition'] = value
                    elif field == 'EQUATION':
                        info['equation'] = value
                        # Extract compound IDs
                        info['compounds'] = re.findall(r'C\d{5}', value)
                    elif field == 'PATHWAY':
                        # Store both ID and name
                        pathway_parts = value.split(None, 1)
                        if len(pathway_parts) == 2:
                            info[field] = [(pathway_parts[0], pathway_parts[1])]
                        else:
                            info[field] = [(value, '')]
                    elif field == 'MODULE':
                        # Store both ID and name
                        module_parts = value.split(None, 1)
                        if len(module_parts) == 2:
                            info[field] = [(module_parts[0], module_parts[1])]
                        else:
                            info[field] = [(value, '')]
                    elif field in ['COMMENT', 'ENZYME', 'ORTHOLOGY']:
                        info[field] = [value]
                    elif field == 'RCLASS':
                        info['RCLASS'] = [value]
                    elif field == 'DBLINKS':
                        info['DBLINKS'] = value
        
        # Clean up enzyme IDs
        if 'ENZYME' in info:
            info['ENZYME'] = [e.strip() for e in ' '.join(info['ENZYME']).split()]
        
        # Clean up orthology IDs
        if 'ORTHOLOGY' in info:
            info['ORTHOLOGY'] = [o.split()[0] for o in info['ORTHOLOGY']]
        
        return info

    def _get_kgml_entry(self, compound_name: str, pathway: str = None):
        """
        Retrieve the KGML entry corresponding to the specified compound name.

        Parameters
        ----------
        compound_name : str
            The name of the compound to search for. The function searches over entries of type "compound"
            and checks if the provided compound name is contained within the entry's name (which may include
            multiple names separated by semicolons) or exactly matches the entry's ID.

        Returns
        -------
        entry
            The KGML entry corresponding to the compound.
        pathway
            if trying to get info from a specific pathway

        Raises
        ------
        AttributeError
            If the KGML pathway has not been loaded.
        ValueError
            If no entry matching the compound name is found.
        """
        if pathway is not None:
            try:
                kgml_to_use = self.pathway_kgmls[pathway]   
            except KeyError:
                self._download_notice(pathway, 'memory for this dataset instance')
                temp_kgml = REST.kegg_get(pathway, 'kgml').read()
                kgml_to_use = KGML_parser.read(StringIO(temp_kgml))
                self.pathway_kgmls[pathway] = kgml_to_use
        else:
            kgml_to_use = self.kgml
        
       
        for entry in kgml_to_use.entries.values():
            if entry.type.lower() == "compound":
                # entry.name may contain multiple compound names separated by a semicolon.
                names = [n.strip() for n in entry.name.split(";")]
                # Return entry if there's an exact match with the compound name or if the compound name is in the list.
                if compound_name in names or compound_name == entry.id:
                    return entry
           
        
        raise ValueError(f"No KGML entry found for compound '{compound_name}'")

    def metabolic_topology(
        self,
        rna: AnnData,
        self_loops: bool = False,
        rebuild_from_kegg: bool = False,
    ) -> nx.DiGraph:
        """
        Return the directed reaction graph.
        
        Parameters
        ----------
        rna
            anndata object
        self_loops
            whether or not to include self loop
        rebuild_from_kegg
            rebuild the graph from the current KEGG KGML instead of using the
            reproducible graph bundled with MERN

        Returns
        --------
        graph
            metabolic topology graph
        """
        if rebuild_from_kegg:
            graph = self._metabolic_topology_from_kgml(rna)
        else:
            graph = self._bundled_graph.copy()

            if not self.add_oxphos:
                graph.remove_nodes_from(self.OXPHOS_RXNS)

            isolates = list(nx.isolates(graph))
            if self.keep_isolates:
                dataset_genes = set(rna.var_names)
                graph.remove_nodes_from(
                    rxn
                    for rxn in isolates
                    if not dataset_genes.intersection(self.rxn_genes.get(rxn, []))
                )
            else:
                graph.remove_nodes_from(isolates)

        nx.set_edge_attributes(graph, 1.0, 'weight')
        nx.set_edge_attributes(graph, 1, 'sign')

        if self_loops:
            for node in graph.nodes:
                graph.add_edge(node, node, weight=1.0, sign=1)

        return graph

    def _metabolic_topology_from_kgml(self, rna: AnnData) -> nx.DiGraph:
        """Build the reaction graph using the downloaded KGML."""
        edge_list = []
        for i in range(len(self.rxns)):
            
            rxn1 = self.rxns[i]
            #reac1 = reactants[i][0]
            prod1 = rxn1.products
            prod1 = [p.name for p in prod1]
            reac1 = rxn1.substrates
            reac1 = [r.name for r in reac1]
            all_comp1 = prod1 + reac1
                
            for j in range(len(self.rxns)):

                rxn2 = self.rxns[j]
                # 01100.kgml lists some reactions twice (distinct entries, same name).
                # Object identity misses those pairs and yields spurious (rn:Rx, rn:Rx) edges.
                if rxn1.name == rxn2.name:
                    continue
                reac2 = rxn2.substrates
                reac2 = [r.name for r in reac2]
                prod2 = rxn2.products
                prod2 = [p.name for p in prod2]
                all_comp2 = prod2 + reac2
                # add both directions
                if len(set(all_comp1).intersection(all_comp2)) > 0:
                    edge_list.append((rxn1.name, rxn2.name))
                    edge_list.append((rxn2.name, rxn1.name))

        graph = nx.DiGraph(edge_list)

        # include all rxns except for isolated rxns with no genes in data
        if self.keep_isolates:
            all_rxn_names = [rxn.name for rxn in self.rxns]
            for rxn in all_rxn_names:
                if rxn in graph.nodes:
                    continue
                else:
                    for gene in self.rxn_genes[rxn]:
                        if gene in rna.var_names:
                            graph.add_node(rxn)
                            break

        return graph

    def compound_metabolic_topology(
        self,
        self_loops: bool = False,
        rxn_weights: pd.Series = None,
        directed: bool = False,
    ) -> nx.DiGraph:
        """
        Composes undirected metabolite-metabolite graph based on metabolic topology.
        
        """
        if directed:
            compound_graph = nx.DiGraph()
        else:
            compound_graph = nx.Graph()
        for reaction in self.rxns:
            
            substrates = [s.name for s in reaction.substrates]
            products = [p.name for p in reaction.products]
            
            # Add edges for each substrate-product pair
            for sub in substrates:
                for prod in products:
                    if rxn_weights is not None:
                        value = rxn_weights.get(reaction.name, 0)
                    else:
                        value = 0
                    compound_graph.add_edge(
                        sub, prod,
                        reaction=reaction.name,
                        value=value,
                    )

        return compound_graph

    def add_module_info(
        self,
        rna: AnnData,
    ):
        """
        Add metabolic module information for each gene in data set
        
        Parameters
        -----------
        rna
            anndata object
        """

        met_g = self.metabolic_topology(rna, self_loops=False)

        gene_rxns = []
        for gene in rna.var_names.to_numpy():

            temp_gene_rxns = []
            for rxn in self.rxn_genes.keys():
                if rxn not in met_g.nodes:
                    continue
                if gene in self.rxn_genes[rxn]:
                    temp_gene_rxns.append(rxn)
                
            gene_rxns.append(','.join(temp_gene_rxns))

        rna.var['Gene Reactions'] = gene_rxns
        rna.var['Metabolic Gene'] = ['Non-metabolic' if rxn=='' else 'Metabolic' for rxn in rna.var['Gene Reactions']]
    
    def add_rxn_module_info(
        self,
        rna: AnnData,
        graph: nx.DiGraph,
    ):
        """
        Add module information on graph nodes to uns
        
        Parameters
        ---------
        rna
            anndata object
        graph
            metabolic topology graph (rxn)
        """
        mods = []
        pathways = []
        pathway_names = []
        broad_classes = []
        narrow_classes = []
        names = []
        rxn_names = []
        for node in graph.nodes:
            node_mods = []
            node_broad_classes = []
            node_narrow_classes = []
            node_names = []
            node_pathways = []
            node_pathway_names = []
            node_rxn_names = []

            for rxn in node.split(' '):
                
                info = self.rxn_info[rxn.replace('rn:', '')]
                try:
                    mod_ids = [mod[0] for mod in info['MODULE']]
                    mod_names = [mod[1] for mod in info['MODULE']]
                except KeyError:
                    mod_ids = []
                    mod_names = []

                node_mods += mod_ids
                node_names += mod_names
                
                try:
                    temp_pathway_ids = [pathway[0] for pathway in info['PATHWAY']]
                    temp_pathway_names = [pathway[1] for pathway in info['PATHWAY']]
                except KeyError:
                    temp_pathway_ids = []
                    temp_pathway_names = []

                node_pathways += temp_pathway_ids
                node_pathway_names += temp_pathway_names

                try:
                    node_rxn_names.append(info['name'])
                except KeyError:
                    node_rxn_names.append('')
        
            mods.append(';'.join(node_mods))
            broad_classes.append(';'.join(node_broad_classes))
            narrow_classes.append(';'.join(node_narrow_classes))
            names.append(';'.join(node_names))
            pathways.append(';'.join(node_pathways))
            pathway_names.append(';'.join(node_pathway_names))
            rxn_names.append(';'.join(node_rxn_names))
        
        rxn_df = pd.DataFrame({"Rxn Name": rxn_names, "Modules":mods, "Broad Classes": broad_classes, "Narrow Classes": narrow_classes, "Names": names, "Pathways": pathways, "Pathway Names": pathway_names}, 
                            index=graph.nodes)
        rna.uns['Reaction Info'] = rxn_df
    
    def get_rxn_genes(
        self,
        rna: AnnData,
        subset_genes: bool = True,
    ):
        """
        Gets dictionary mapping rxns to genes

        
        Parameters
        ----------
        rna
            RNA dataset
        subset_genes
            subset to only genes in data set
    
        Returns
        --------
        rxn_genes
            Mapping from rxns to genes
        """
        rxn_gene_symbols = {rxn: list(genes) for rxn, genes in self.rxn_genes.items()}

        if not subset_genes:
            return rxn_gene_symbols

        dataset_genes = set(rna.var_names)

        # filter rxn genes based on genes in dataset
        for rxn in rxn_gene_symbols:
            temp_genes = set(rxn_gene_symbols[rxn])
            final_genes = temp_genes.intersection(dataset_genes)
            rxn_gene_symbols[rxn] = list(final_genes)

        return rxn_gene_symbols
    
    def flatten_rxns(self):
        """
        Args:
            graphs (dict): dictionary containing edge lists of graphs for each module
        Returns:
            all_rxns (list): list of all included rxns
        """
        raise NotImplementedError
    
    def calculate_stoichiometric_mat(
        self
    ) -> pd.DataFrame:
        """
        Calculates stoichiometric matrix

        Parameters
        ----------
        species
            species to use

        Returns
        -------
        stoichiometric matrix as df
        """
        raise NotImplementedError
    
    def calculate_metabolites(
        self,
        rna: AnnData,
        absolute: bool = False,
    ):
        """
        Calculates relative metabolite activations based on rxn activations. Modifies anndata object

        Parameters
        ---------
        rna
            anndata object with module info
        
        Returns
        --------
        met_df
            df with cells by metabolites
        """
        raise NotImplementedError
