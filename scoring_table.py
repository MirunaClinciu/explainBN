from .utilities import desextremize, factor_argmax
from .utilities import get_factor_from_scope, create_trivial_argument, get_adjacent_factors_scope, get_endpoints, is_endpoint, factor_distance

from itertools import product
from collections import defaultdict, namedtuple

import numpy as np

# Link contribution
def compute_link_contribution(model, factor, incoming_messages, target):
  other_nodes = set(factor.variables) - {target}

  f1 = reduce(lambda a,b: a.product(b, inplace=False),
              incoming_messages.values(),
              factor)
  f2 = factor.copy()

  f1.normalize()
  f2.normalize()

  f1.marginalize(other_nodes)
  f2.marginalize(other_nodes)

  f1.normalize()
  f2.normalize()

  f1.values = desextremize(f1.values)
  f2.values = desextremize(f2.values)
  
  delta = f1.divide(f2, inplace=False)
  delta.normalize()

  assert set(delta.scope()) == {target}, delta.scope()

  return delta

# Argument importance computation

def evidence_levels(logodd_score):
  if logodd_score > 2:
    return "extremely"
  elif logodd_score > 1:
    return "greatly"
  elif logodd_score > 0.5:
    return "moderately"
  else: #if logodd_score > -0.00001:
    return "slightly"
  
  assert False, "The logodd score should not be negative"

def explain_delta(model, delta):
  # Find state favored by evidence
  delta = delta.normalize(inplace=False)
  arg = factor_argmax(delta)[0]

  # Convert delta to logodds
  p = delta.get_value(**dict([arg]))
  s = np.log(p / (1. - p))

  # Craft explanation
  if model.explanation_dictionary[arg]['polarity'] == 'positive':
    explanation = model.explanation_dictionary[arg]['explanation']
    verb = 'increases'
  else:
    verb = 'decreases'
    explanation = model.explanation_dictionary[arg]['contrastive_explanation']
  
  evidence_level = evidence_levels(s)
  
  return f"this {evidence_level} {verb} the likelihood that {explanation}"

def get_delta(model, argument, node):
  
  # If the node is and endpoint, return a fresh delta
  if node in argument.observations: 
    obs_explanation = model.explanation_dictionary[((node, argument.observations[node]))]['explanation']
    explanation = [f"We have observed that {obs_explanation}"]
    return init_delta(model, node, argument.observations[node]), explanation

  factor_scope = next(argument.successors(node))
  factor = get_factor_from_scope(model, factor_scope)

  incoming_messages = {}
  explanation = []
  buffer = []
  first = True
  for child in argument.successors(factor_scope):

    if not first:
      pass
      #buffer.append("since " + child_explanation[-1])
    
    incoming_messages[child], child_explanation = \
      get_delta(model, argument, child)
    
    # print(child_explanation)

    # Improve readability of explanation 
    # when the argument splits off multiple ways
    if not first:
      child_explanation[0] = "on the other hand, " \
                           + child_explanation[0].lower()
    
    explanation += child_explanation
    first = False
  
  # Add buffer to remind user of context
  explanation += buffer

  delta = compute_link_contribution(model, factor, incoming_messages, node)
  delta_explanation = explain_delta(model, delta)
  explanation.append(delta_explanation)

  return delta, explanation

def compute_argument_importance(model, argument, target):
  # Compute argument delta
  delta, explanation = get_delta(model, argument, target[0])

  # Convert delta to logodds depending on target
  delta.normalize()
  p = delta.get_value(**dict([target]))
  s = np.log(p / (1. - p))

  return s, explanation

#@title Interaction detector

def find_d_interactions(model, factor, target, th = 1.0):
  other_nodes = set(factor.variables) - {target}
  unmodularizable_interactions = set()
  contribution_map = {}
  # Iterate over possible sources
  for r in range(1, len(other_nodes)+1):
    for sourceset in combinations(other_nodes, r):
      # Instantiate the evidence
      instantiations = product(*[[(node, value) 
                                  for value in model.states[node]]
                                 for node in sourceset])
      for sourceset_instantiation in instantiations:
        sourceset_instantiation = frozenset(sourceset_instantiation)
        incoming_messages = {node : init_delta(model, node, state)
                             for node, state in sourceset_instantiation}

        contribution = compute_link_contribution(model, 
                                                 factor, 
                                                 incoming_messages,
                                                 target
                                                )
        
        contribution_map[sourceset_instantiation] = contribution

        # Atomic contributions between parents and child are always considered
        if r == 1:
          child = get_child_from_factor_scope(model, factor.scope())
          if sourceset[0] in child or target in child:
            unmodularizable_interactions.add(sourceset)
          continue

        # Check if the contribution can be modularized
        for obs in sourceset_instantiation:
          other = sourceset_instantiation - {obs}
          interaction = contribution_map[sourceset_instantiation]
          p1 = contribution_map[frozenset([obs])]
          p2 = contribution_map[other]
          combination = p1.product(p2, inplace=False)
          d = factor_distance(interaction, combination)
          # print(d)
          if d < th:
            break
        else:
          unmodularizable_interactions.add(sourceset)
      
  return unmodularizable_interactions

#@title Argument extender

def extend_argument(model, argument):
  # Dynamically find extension combinations that cannot be modularized
  unmodularizable_extensions = defaultdict(set)

  for endpoint in get_endpoints(argument):
    # Add default extension
    unmodularizable_extensions[endpoint].add(None)

    # Find adjacent factors in the factor graph
    prev_factor_scope = list(argument.predecessors(endpoint))[0] \
                        if argument.in_degree(endpoint) > 0 else None
    adj_factors_scope = \
       set(get_adjacent_factors_scope(model, endpoint)) \
       - {prev_factor_scope}

    for factor_scope in adj_factors_scope:
      factor = get_factor_from_scope(model, factor_scope)
      extensions = find_d_interactions(model, factor, endpoint)
      # Add all extensions, except those that would 
      # require adding a node which is already in the argument
      # and it is not an endpoint
      unmodularizable_extensions[endpoint] |= {
         (factor_scope, extension) for extension in extensions \
         if np.all(list(map(
                       lambda node: (is_endpoint(argument, node) \
                                    or node not in argument.nodes()),
                       extension)))
         }

  # print(unmodularizable_extensions)
  
  # Construct possible extensions
  extended_arguments = []
  possible_extensions = product(*[[(endpoint, extension) 
                                    for extension in unmodularizable_extensions[endpoint]]
                                    for endpoint  in unmodularizable_extensions])
  
  for possible_extension in possible_extensions:
    possible_extension = dict(possible_extension)
    extended_argument = argument.copy()
    has_been_extended = False

    for endpoint, extension in possible_extension.items():
      if extension is None: continue
      factor_scope, nodes = extension
      extended_argument.add_edge(endpoint, factor_scope)
      for node in nodes:
        extended_argument.add_edge(factor_scope, node)
      has_been_extended = True

    # We note down the argument if at least one extension has been applied
    # and no cycles have been introduced by the extension
    if has_been_extended and \
       len(list(nx.simple_cycles(extended_argument))) == 0:
      extended_arguments.append(extended_argument)

  return extended_arguments

# Argument instantiator
def instantiate_argument(model, argument):
  endpoints = set(get_endpoints(argument))
  possible_instantiations = product(*[[(node, value) 
                                      for value in model.states[node]]
                                     for node in endpoints])
  instantiated_arguments = []

  for instantiation in possible_instantiations:
    instantiated_argument = argument.copy()
    # instantiated_argument.target = argument.target
    instantiated_argument.observations = dict(instantiation)
    instantiated_arguments.append(instantiated_argument)

  return instantiated_arguments

# Scoring table construction
def generate_scoring_table(model, 
                           target, 
                           evidence_nodes, 
                           th=0.2):
  # Retrieve baseline logodds
  baseline_p = model.baselines[target[0]].get_value(**{target[0]:target[1]})
  baseline_logodds = np.log(baseline_p / (1.-baseline_p))

  # Utility to consistently hash arguments
  NodesAndEdges = namedtuple('NodesAndEdges', ['nodes', 'edges'])
  argument_to_nae = lambda arg : NodesAndEdges(frozenset(arg.nodes), 
                                              frozenset(arg.edges))

  # Find argumentative trees to target  
  trivial_argument = create_trivial_argument(target[0])
  arguments_to_explore = extend_argument(model, trivial_argument)
  arguments_explored = set()
  relevant_arguments = []

  while len(arguments_to_explore) > 0:
    argument = arguments_to_explore.pop()
    
    nae = argument_to_nae(argument)
    if nae in arguments_explored: continue

    #print(nae)

    arguments_explored.add(nae)
    is_argument_relevant = False

    instantiated_arguments = instantiate_argument(model,argument)
    for instantiated_argument in instantiated_arguments:

      argument_score, explanation = \
        compute_argument_importance(model, 
                                    instantiated_argument, 
                                    target)
      
      # If the score associated with the instantiated argument
      # is over the threshold we add it to the table and expand it later
      if np.abs(argument_score) > th:
        relevant_arguments.append({
            "argument": instantiated_argument,
            "explanation": explanation,
            "score" : argument_score,
            "observations": instantiated_argument.observations})
        is_argument_relevant = True
    
    if is_argument_relevant:
      # Add expansions of argument to the stack
      for extended_argument in extend_argument(model, argument):
        # if extended_argument in arguments_explored: continue
        arguments_to_explore.append(extended_argument)

  # Create scoring table
  scoring_table = pd.DataFrame(relevant_arguments)

  # Sort scoring table according to absolute score contributions
  scoring_table = scoring_table.loc[(scoring_table["score"].abs())\
                                    .sort_values(ascending=False).index]\
                                    .reset_index(drop=True)

  # Add baseline entry on top of the table
  target_explanation = model.explanation_dictionary[target]["explanation"]
  scoring_table.loc[-1] = \
    {
      "observation" : f"({target[0]},{target[1]}) (baseline)",
      "explanation" : f"The baseline probability that {target_explanation} is {baseline_p*100:0.2f}%",
      "score" : baseline_logodds,
      "argument" : trivial_argument,
      "observations": {},
    }
  scoring_table.index = scoring_table.index + 1
  scoring_table.sort_index(inplace=True)

  return scoring_table


