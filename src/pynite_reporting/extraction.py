import math
from typing import Optional, Union, Any
import numpy as np


# ACTIONS = ['Fy', 'Fz', 'Mz', 'My', 'axial', 'torque']


# ACTIONS_BY_TYPE = {
#         "shear": ["Fy", "Fz"], # action: [possible direction(s)]
#         "moment": ["Mz", "My"],
#         "axial": ["axial"], # There are no axial directions; axial is axial!
#         "torque": ["torque"], # ...same with torque
#         "deflection": ['dx', 'dy', 'dz']
# }

ACTION_METHODS = {
    "Fy": "shear",
    "Fz": "shear",
    "My": "moment",
    "Mz": "moment",
    "axial": "axial",
    "torque": "torque",
    "dy": "deflection",
    "dz": "deflection",
    "dx": "deflection",
}

NODE_REACTIONS = ['FX', 'FY', 'FZ', 'MX', 'MY', 'MZ']
NODE_DISPLACEMENTS = ['DX', 'DY', 'DZ', 'RX', 'RY', 'RZ']

REACTIONS = ['RxnFX', 'RxnFY', 'RxnFZ', 'RxnMX', 'RxnMY', 'RxnMZ']


def extract_reactions(
    model: "Pynite.FEModel3D", 
    load_combinations: Optional[list[str]] = None,
    results_key: Optional[str] = "node_reactions",
) -> dict[str, dict]:
    if load_combinations is None:
        load_combinations = extract_load_combinations(model)

    reaction_results = {}
    # Go through all the nodes...
    for node_name, node in model.nodes.items():
        reaction_results.setdefault(node_name, {})
        # ...and go through all reaction directions...
        inner_acc = reaction_results[node_name]
        if results_key is not None:
            reaction_results[node_name].setdefault(results_key, {})
            inner_acc = reaction_results[node_name][results_key]
        for reaction_dir in NODE_REACTIONS:
            reaction_name = f"Rxn{reaction_dir}"
            # Get the reactions...
            reactions = getattr(node, reaction_name)
            # ...and if the reactions are not all basically 0.0...
            if not (
                all([math.isclose(reaction, 0, abs_tol=1e-8) for reaction in reactions.values()])
            ):
                # Then collect them in our analysis results dictionary
                inner_acc[reaction_dir] = {
                    lc: round_to_close_integer(reaction) 
                    for lc, reaction in reactions.items()
                    if lc in load_combinations
                }
        # But if any of the nodes in the analysis results dict are empty...
        if reaction_results[node_name] == {}:
            # ...then drop 'em!
            reaction_results.pop(node_name)
    return reaction_results


def extract_node_deflections(
    model: "Pynite.FEModel3D", 
    load_combinations: Optional[list[str]] = None,
    results_key: Optional[str] = "node_deflections",
):
    if load_combinations is None:
        load_combinations = extract_load_combinations(model)
    node_deflections = {}
    # Go through all the nodes...
    for node_name, node in model.nodes.items():
        node_deflections[node_name] = {}

        inner_acc = node_deflections[node_name]
        if results_key is not None:
            node_deflections[node_name].setdefault(results_key, {})
            inner_acc = node_deflections[node_name][results_key]
        # ...and go through all deflection directions...
        for defl_dir in NODE_DISPLACEMENTS:
            # Get the deflections...
            deflections = getattr(node, defl_dir)
            # ...and if the deflections are not all basically 0.0...
            if not (
                all([math.isclose(deflection, 0, abs_tol=1e-8) for deflection in deflections.values()])
            ):
                # Then collect them in our analysis results dictionary
                inner_acc[defl_dir] = {
                    lc: float(defl) 
                    for lc, defl in deflections.items()
                    if lc in load_combinations
                }

        # But if any of the nodes in the analysis results dict are empty...
        if node_deflections[node_name] == {}:
            # ...then drop 'em!
            node_deflections.pop(node_name)

    return node_deflections


def extract_member_force_arrays(
    model: "Pynite.FEModel3D", 
    load_combinations: Optional[list[str]] = None,
    n_points: int = 1000,
    results_key: Optional[str] = "frame_arrays"
) -> dict[str, dict]:
    """
    
    """
    forces = {}
    # For each member...
    if load_combinations is None:
        load_combinations = extract_load_combinations(model)
    forces = {}
    for member_name, member in model.members.items():
        # For each action...
        forces[member_name] = {}
        inner_acc = forces[member_name]
        if results_key is not None:
            forces[member_name].setdefault(results_key, {})
            inner_acc = forces[member_name][results_key]
        for force_direction, method_type in ACTION_METHODS.items():
            inner_acc.setdefault(force_direction, {})
            array_method = getattr(member, f"{method_type}_array")
            accumulator = inner_acc[force_direction]
            parent_accumulator = inner_acc
            path = force_direction

            # if force_direction == method_type:
            #     accumulator = inner_acc[action_name]

            #     # The 'parent_accumulator' and 'path' allow me to update the action
            #     # with None if there are no results
            #     parent_accumulator = inner_acc
            #     path = action_name
            # else:
            #     inner_acc[action_name][direction] = {}
            #     accumulator = inner_acc[action_name][direction]

            #     # The 'parent_accumulator' and 'path' allow me to update the action
            #     # with None if there are no results
            #     parent_accumulator = inner_acc[action_name]
            #     path = direction
    
            # ...and for each load combo...
            for load_combo_name in load_combinations:
                accumulator.setdefault(load_combo_name, {})
                
                # pop_next = False

                # hacky fix: AxialDeflection method is receiving None for a .P1, I think (Pynite error to be fixed)
                try:
                    if method_type not in ("axial", "torque"):
                        result_arrays = array_method(force_direction, combo_name=load_combo_name, n_points=n_points)
                    else:
                        result_arrays = array_method(combo_name=load_combo_name, n_points=n_points)
                except TypeError:
                    parent_accumulator[path] = None
                    continue

                if (
                    (result_arrays.dtype == "object") # An array with None (or full of None_s)
                    or
                    np.allclose(result_arrays[1], np.zeros(len(result_arrays[1])), atol=1e-8)
                ):
                    accumulator.pop(load_combo_name)
                    pass
                else:
                    accumulator[load_combo_name] = result_arrays
                # if pop_next:
                #     parent_accumulator.pop(path)
            if not parent_accumulator[path]:
                parent_accumulator.pop(path)
    return forces


def extract_member_forces_minmax(
        model: "Pynite.FEModel3D", 
        load_combinations: Optional[list[str]] = None,
        results_key: Optional[str] = "frame_envelopes"
    ) -> dict[str, dict]:

    # For each member...
    if load_combinations is None:
        load_combinations = extract_load_combinations(model)
    forces = {}
    for member_name, member in model.members.items():
        forces[member_name] = {}
        inner_acc = forces[member_name]
        if results_key is not None:
            forces[member_name].setdefault(results_key, {})
            inner_acc = forces[member_name][results_key]
        for force_direction, method_type in ACTION_METHODS.items():
            inner_acc[force_direction] = {}
            max_method = getattr(member, f"max_{method_type}")
            min_method = getattr(member, f"min_{method_type}")
            accumulator = inner_acc[force_direction]
            parent_accumulator = inner_acc
            path = force_direction

            for load_combo_name in load_combinations:
                accumulator[load_combo_name] = {}
                
                if method_type not in ("axial", "torque"):
                    max_value = float(max_method(force_direction, load_combo_name))
                    min_value = float(min_method(force_direction, load_combo_name))
                else:
                    max_value = float(max_method(load_combo_name))
                    min_value = float(min_method(load_combo_name)) 
                abs_max_value = max([abs(min_value), abs(max_value)])
                if math.isclose(max_value, 0, abs_tol=1e-8):
                    max_value = 0.
                if math.isclose(min_value, 0, abs_tol=1e-8):
                    min_value = 0.

                if min_value == max_value == 0.:
                    accumulator.pop(load_combo_name)
                    pass
                else:
                    accumulator[load_combo_name].update({f"max": round_to_close_integer(max_value)})
                    accumulator[load_combo_name].update({f"min": round_to_close_integer(min_value)})
                    accumulator[load_combo_name].update({f"absmax": round_to_close_integer(abs_max_value)})
            if not parent_accumulator[path]:
                parent_accumulator.pop(path)
    return forces
        

def extract_member_forces_at_locations(
    model: "Pynite.FEModel3D", 
    force_extraction_locations: Optional[dict[str, list[float]]] = None,
    force_extraction_ratios: Optional[dict[str, list[float]]] = None,
    load_combinations: Optional[list[str]] = None,
    by_span: bool = False,
    results_key: Optional[str] = "frame_force_locations"
) -> dict[str, dict]:
    """
    Extracts forces at selected members at the locations specified.

    'force_extraction_locations': a dict in the following format:

        {"member01": [0, 2200, 4300], "member02": [3423, 1500]}

        Where:
        - "member01" is a member name
        - The values are a list of locations on the member from which to
            extract results from.

    'force_extraction_ratios': a dict in the following format:

            {
                "member01": [0.25, 0.5, 0.77], 
                "member02": [0.333, 0.666],
                ...    
            }

        Where:
        - "member01" is a member name
        - The values are a list of ratios on the member from which to
            extract results from. The location is calculated by 

            ratio * member.L() # length

            Whether the member is the PhysMember3D (main member)
            or a Member3D (sub member, i.e. an individual span).
    """
    if load_combinations is None:
        load_combinations = extract_load_combinations(model)
    if force_extraction_locations is None:
        force_extraction_locations = {}
    if force_extraction_ratios is None:
        force_extraction_ratios = {}
    
    force_locations = {}
    for member_name, member in model.members.items():
        force_locations.setdefault(member_name, {})
        inner_acc = force_locations[member_name]
        if results_key is not None:
            force_locations[member_name].setdefault(results_key, {})
            inner_acc = force_locations[member_name][results_key]
        if member_name not in (
            list(force_extraction_locations.keys()) 
            + list(force_extraction_ratios.keys())
        ):
            continue
        if by_span:
            inner_acc.setdefault("by_span": [])
            for sub_member in member.sub_members.values():
                inner_acc['by_span'].append(
                    collect_forces_at_location(
                        sub_member,
                        member_name,
                        force_extraction_locations, 
                        force_extraction_ratios, 
                        load_combinations
                    )
                )
        else:
            inner_acc = collect_forces_at_location(
                member,
                member_name,
                force_extraction_locations,
                force_extraction_ratios,
                load_combinations
            )
    return force_locations


def collect_forces_at_location(
    submember: "Pynite.Member3D",
    member_name: str,
    force_extraction_locations: dict, 
    force_extraction_ratios: dict,
    load_combinations: list[str]
) -> dict:
    acc = {}
    for loc in force_extraction_locations.get(member_name,{}):
        acc.update({loc: extract_forces_at_location(submember, loc, load_combinations)})

    for ratio in force_extraction_ratios.get(member_name, {}):
        length = submember.L()
        loc = length * ratio
        acc.update({loc: extract_forces_at_location(submember, loc, load_combinations)})
    return acc


def extract_forces_at_location(member: "Pynite.Member3D", location: float, load_combinations: list[str]):
    loc = location
    acc = {}
    for load_combo_name in load_combinations:
        acc[load_combo_name] = {}
        for force_direction, method_type in ACTION_METHODS.items():
            force_method = getattr(member, method_type)
            if force_method not in ("axial", "torque"):
                force_value = round_to_close_integer(force_method(force_direction, loc, load_combo_name))
            else:
                force_value = round_to_close_integer(force_method(loc, load_combo_name))
            acc[load_combo_name].update({force_direction: force_value})
    return acc


def extract_span_forces_minmax(
    model: "Pynite.FEModel3D", 
    load_combinations: Optional[list[str]] = None, 
    actions: Optional[list[str]] = None,
    results_key: Optional[str] = "frame_span_envelopes",
) -> dict:
    """
    Returns a dict of the following shape which represents the results extract from each span of
    each member in 'model':

        {
            "member": {
                "result_key": {
                    "LC": {
                        "Action": { # Where Action is one of My, Mz, Fy, Fz, axial, torque, dx, dy 
                            "max": [
                                {"value": Yi, "loc_rel": xi, "span_length": li, "loc_abs": Xi, "span": Li},
                                ...
                            ],
                            "min": [
                                {"value": Yi, "loc_rel": xi, "span_length": li, "loc_abs": Xi, "span": Li},
                                ...
                            ]
                        },

                    },
                },
            },
        }
    'load_combinations': If provided, will only extract these load combinations (extract all if None)
    'actions': If provided, will only extract these actions (extract all if None)
        possible actions: {'Fy', 'Fz', 'My', 'Mz', 'axial', 'torque', 'dy', 'dx'}
    """
    if actions is None:
        actions = ['Fy', 'Fz', 'My', 'Mz', 'axial', 'torque', 'dy', 'dx']
    max_min = ['max', 'min']
    if load_combinations is None:
        load_combinations = extract_load_combinations(model)
    n_points = 1000
    member_spans = extract_spans(model)
    span_envelopes = {}
    for member_name, sub_members in member_spans.items():
        member_length = model.members[member_name].L()
        span_envelopes.setdefault(member_name, {})
        for lc in load_combinations:
            span_envelopes[member_name].setdefault(lc, {})
            for force_direction, method_type in ACTION_METHODS.items():
                span_envelopes[member_name][lc].setdefault(force_direction, {})
                method_name = f"{method_type}_array"
                for envelope in max_min:
                    envelope_key = f"span_envelope_{envelope}"
                    span_envelopes[member_name][lc][force_direction].setdefault(envelope_key, [])
                    length_counter = 0
                    for sub_member in sub_members:
                        method = getattr(sub_member, method_name)
                        if method_type not in ('axial', 'torque'):
                            result_arrays = method(force_direction, n_points=n_points, combo_name=lc)
                        else:
                            result_arrays = method(n_points=n_points, combo_name=lc)
                        locator_func = getattr(np, f'arg{envelope}')
                        envelope_func = getattr(np, envelope)
                        envelope_val = envelope_func(result_arrays[1])
                        envelope_idx = locator_func(result_arrays[1])
                        x_val_local = result_arrays[0][envelope_idx]
                        x_val_global = x_val_local + length_counter
                        x_length = result_arrays[0][-1]
                        length_counter += x_length
                        is_cantilevered = member_is_cantilevered(sub_member)
                        span_envelope = {
                            "value": round_to_close_integer(envelope_val),
                            "loc_rel": x_val_local,
                            "span_length": sub_member.L(),
                            "loc_abs": x_val_global,
                            "length": member_length,
                            "is_cantilever": is_cantilevered
                        }
                        span_envelopes[member_name][lc][force_direction][envelope_key].append(span_envelope)
    return span_envelopes


def extract_spans(model: "Pynite.FEModel3D") -> dict[str, list["Pynite.Member3D"]]:
    """
    Extracts the sub-members for all of the members in the 'model'
    """
    member_spans = {}
    for member_name, member in model.members.items():
        member_spans.setdefault(member_name, [])
        for name, span_member in model.members[member_name].sub_members.items():
            member_spans[member_name].append(span_member)
    return member_spans


def extract_load_combinations(model: "Pynite.FEModel3D") -> list[str]:
    """
    Returns a list of the load combination names used in the model
    """
    return list(model.load_combos.keys())


def member_is_cantilevered(member: "Pynite.Member3D") -> bool:
    """
    Returns True if a member is cantilevered. False otherwise.
    """
    has_two_supports = member_has_two_supports(member)
    if has_two_supports:
        return False
    return member_has_reactions_each_end(member)


def member_has_two_supports(member: "Pynite.Member3D") -> bool:
    """
    Returns True if 'member' two supports.
    False if it has less than two supports.
    """
    return all([
        node_has_supports(member.i_node),
        node_has_supports(member.j_node),
    ])


def member_has_reactions_each_end(member: "Pynite.Member3D") -> bool:
    """
    Returns True if the 'member' has at least one reaction one both
    ends.
    False, otherwise.
    """
    reactions_i_tally = []
    reactions_j_tally = []
    for reaction_type in REACTIONS:
        i_node = member.i_node
        j_node = member.j_node
        reactions_i = getattr(i_node, reaction_type)
        reactions_j = getattr(j_node, reaction_type)
        reactions_i_tally.append(any([round_to_close_integer(reaction) for reaction in reactions_i.values()]))
        reactions_j_tally.append(any([round_to_close_integer(reaction) for reaction in reactions_j.values()]))
    return any([reactions_i_tally]) and any([reactions_j_tally])

def node_has_supports(node: "Pynite.Node") -> bool:
    """
    Returns True if 'node' has any supports defined on it.
    False if it is a "free" node.
    """
    return any([
        node.support_DX,
        node.support_DY,
        node.support_DZ,
        node.support_RX,
        node.support_RY,
        node.support_RZ
    ])


def round_to_close_integer(x: float, eps = 1e-8) -> float | int:
    """
    Rounds to the nearest int if it is REALLY close
    """
    if abs(abs(round(x)) - abs(x)) < eps:
        return round(x)
    else:
        return x
    

