import numpy as np
from itertools import product, combinations, chain, tee, islice

from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import DiscreteFactor

# GENERAL UTILITIES
def powerset(iterable):
    """
    powerset([1,2,3]) --> 
    () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def nwise(iterable, n=2):
    iters = tee(iterable, n)
    for i, it in enumerate(iters):
        next(islice(it, i, i), None)
    return zip(*iters)

  
from_prob_to_logodd = lambda p : np.log(p / (1. - p))
from_logodd_to_prob = lambda s : 1. / (1. + 1. / np.exp(s))

# PGMPY UTILITIES
prob = lambda model, target, evidence : \
          VariableElimination(model).query(variables=[target[0]], 
                                           evidence=evidence, 
                                           show_progress=False)\
          .get_value(**{target[0]:target[1]})

def partial_normalize(phi, scope):
  phi1 = phi.copy()
  complementary_scope = set(phi.variables) - set(scope)
  possible_neg_scope_values = product(*[[(node, value) 
                                  for value in factor.state_names[node]]
                                  for node in complementary_scope])
  
  for neg_scope_values in possible_neg_scope_values:
    f = phi1.reduce(neg_scope_values, inplace=False)
    f.normalize()

    possible_scope_values = product(*[[(node, value) 
                                  for value in factor.state_names[node]]
                                  for node in scope])
    for scope_values in possible_scope_values:
      v = f.get_value(**dict(scope_values))
      phi1.set_value(v, **dict(scope_values), **dict(neg_scope_values))
  
  return phi1

def reciprocal_factor(phi):
  phi = phi.copy()
  phi.values = 1./phi.values
  phi.normalize()
  return phi

def factor_argmax(factor):
  possible_scope_values = product(*[[(node, value) 
                                  for value in factor.state_names[node]]
                                  for node in factor.variables])
  max_value = 0.
  max_scope = None
  for scope_value in possible_scope_values:

    v = factor.get_value(**dict(scope_value))
    if v > max_value:
      max_value = v
      max_scope = scope_value
  return max_scope

def init_delta(model, node, observed_state, 
               eps = 0.0000001):
  """ Initializes a factor to a extreme state 
      corresponding to an observation
  """
  possible_states = model.states[node]
  m = [1. if state == observed_state else eps
       for state in possible_states]
  delta = DiscreteFactor([node], 
                         [len(possible_states)], 
                         [m], 
                         {node:possible_states})
  delta.normalize()
  return delta

def init_uninformative_prior(model, node):
  """ Initializes a factor to a maximum entropy state
  """
  possible_states = model.states[node]
  m = [1. for state in possible_states]
  delta = DiscreteFactor([node], 
                         [len(possible_states)], 
                         [m], 
                         {node:possible_states})
  delta.normalize()
  return delta

eps = 0.0000001
desextremize = np.vectorize(lambda f : 
                            eps if f < eps 
                            else 1. - eps 
                            if f > 1. - eps 
                            else f)

# ARGUMENT UTILITIES

def make_argument_from_chain(chain):
  argument = nx.DiGraph()
  iter_nodes = iter(chain)
  first_node = next(iter_nodes)
  argument.add_node(first_node)
  prev_node = first_node
  for node in iter_nodes:
    argument.add_edge(prev_node, node)
    prev_node = node
  return argument

def factor_distance(f1, f2):
  delta = f1.divide(f2, inplace = False)
  delta.normalize()
  max_delta = 0.0
  for node in delta.variables:
    for state in delta.state_names[node]:
      p = delta.get_value(**dict([(node, state)]))
      s = np.log(p / (1. - p))
      if np.abs(s) > max_delta: max_delta = np.abs(s)
  return max_delta

def get_child_from_factor_scope(model, factor_scope):
  for node1 in factor_scope:
    for node2 in factor_scope:
      if node1 != node2 and node2 in model.get_children(node1):
        break
    else:
      return node1

def create_trivial_argument(node):
  argument = nx.DiGraph()
  argument.add_node(node)
  return argument

def get_endpoints(argument):
  return [node for node in argument.nodes if argument.out_degree(node) == 0]

def is_endpoint(argument, node):
  return argument.out_degree(node) == 0

def get_factor_from_scope(model, scope):
  child = get_child_from_factor_scope(model, scope)
  f = model.get_cpds(child).to_factor()
  return f

def get_adjacent_factors_scope(model, node):
  upper_factor = frozenset(model.get_cpds(node).scope())
  lower_factors = frozenset(frozenset(model.get_cpds(child).scope()) 
                   for child in model.get_children(node))
  return frozenset([upper_factor]) | lower_factors

# DRAWING GRAPHS WITH HIGHLIGHTED PATHS
swap = lambda edge : (edge[1], edge[0])

def get_target(argument):
  for node in argument.nodes:
    if argument.in_degree(node) == 0:
      return node
  
  assert False, "The argument loops on itself and has no conclusion"

def draw_model(model, arguments=[], argument=None):
  G = model
  df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
  for row, data in nx.shortest_path_length(G):
    for col, dist in data.items():
        df.loc[row,col] = dist
  df = df.fillna(df.max().max())
  pos = nx.kamada_kawai_layout(G,dist=df.to_dict())

  if argument is not None:
    arguments = [argument]
  
  if len(arguments) > 0:
    nodes_to_color = []
    edges_to_color = []
    sources = []
    
    for argument in arguments:
      nodes_to_color += [node for node in argument if node in model]
      edges_and_swapped_edges = list(argument.to_undirected().edges) + \
                                [(n2,n1) for (n1,n2) in argument.to_undirected().edges]
      edges_to_color += [(node1, node2) 
                         for (node1, factor1) in edges_and_swapped_edges 
                         for (factor2, node2) in edges_and_swapped_edges
                         if isinstance(node1, str) and isinstance(node2, str) 
                         and node1 != node2 and factor1 == factor2]
      sources += get_endpoints(argument)
      
    target = get_target(argument)

    node_color_map = ['orange' if node == target else
                      'blue' if node in sources else 
                      'green' if node in nodes_to_color else 
                      'gray' 
                      for node in model]
        
    edge_color_map = ['green' if edge in edges_to_color 
                              or swap(edge) in edges_to_color 
                      else 'gray' 
                      for edge in model.edges]
  else:
    node_color_map = 'yellow'
    edge_color_map = 'green'

  nx.draw(G, pos, 
          node_size = 800,
          width = 3,
          with_labels=True, 
          node_color = node_color_map,
          edge_color = edge_color_map,
          font_weight = 'bold')
