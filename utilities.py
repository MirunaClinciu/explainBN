from itertools import product, combinations, chain, tee, islice

import numpy as np
import networkx as nx

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
