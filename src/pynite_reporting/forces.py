import math
from typing import Optional, Union, Any
import numpy as np

def extract_reactions(model: "Pynite.FEModel3D") -> dict[str, dict]:
    reactions = {}
    # Go through all the nodes...
    for node_name, node in model.nodes.items():
        reactions[node_name] = {}
        # ...and go through all reaction directions...
        for reaction_dir in ['FX', 'FY', 'FZ']:
            reaction_name = f"Rxn{reaction_dir}"
            # Get the reactions...
            reactions = getattr(node, reaction_name)
            # ...and if the reactions are not all basically 0.0...
            if not (
                all([math.isclose(reaction, 0, abs_tol=1e-8) for reaction in reactions.values()])
            ):
                # Then collect them in our analysis results dictionary
                reactions[node_name][reaction_dir] = {lc: float(reaction) for lc, reaction in reactions.items()}
        # But if any of the nodes in the analysis results dict are empty...
        if reactions[node_name] == {}:
            # ...then drop 'em!
            reactions.pop(node_name)

    return reactions


def extract_node_deflections(model: "Pynite.FEModel3D", load_combinations: Optional[list[str]] = None)

    node_deflections = {}

    # Go through all the nodes...
    for node_name, node in model.nodes.items():
        node_deflections[node_name] = {}

        # ...and go through all deflection directions...
        for defl_dir in ['DX', 'DY', 'DZ', 'RX', 'RY', 'RZ']:
            # Get the deflections...
            deflections = getattr(node, defl_dir)
            # ...and if the deflections are not all basically 0.0...
            if not (
                all([math.isclose(deflection, 0, abs_tol=1e-8) for deflection in deflections.values()])
            ):
                # Then collect them in our analysis results dictionary
                node_deflections[node_name][defl_dir] = {lc: float(defl) for lc, defl in deflections.items()}

        # But if any of the nodes in the analysis results dict are empty...
        if node_deflections[node_name] == {}:
            # ...then drop 'em!
            node_deflections.pop(node_name)

    return node_deflections




def extract_member_forces_minmax(
        model: "Pynite.FEModel3D", 
        load_combinations: Optional[list[str]] = None
    ) -> dict[str, dict]:
    actions = {
        "shear": ["Fy", "Fz"], # action: [possible direction(s)]
        "moment": ["Mz", "My"],
        "axial": ["axial"], # There are no axial directions; axial is axial!
        "torque": ["torque"], # ...same with torque
    }

    # There many things we need to specify in order to get a number:
    # member_name -> 'forces' -> 'moment'/'shear'/'axial'/'torque' -> Optional[direction] -> load_combo -> 'max'/'min'
    # Which is why there is a loop *four* layers deep
    #
    # This is also complicated by the fact that Pynite subcategorizes forces into
    # "shear", "moment", "axial", and "torque". "shear" and "moment" require an additional
    # direction specification to retrieve the value whereas "axial" and "torque" do not.

    # For each member...
    forces = {}
    for member_name, member in model.members.items():
        # For each action...
        forces[member_name] = {}
        for action_name, directions in actions.items():
            forces[member_name][action_name] = {}
            # ...and for each direction...
            for direction in directions:
                max_method = getattr(member, f"max_{action_name}")
                min_method = getattr(member, f"min_{action_name}")
                if action_name == direction:
                    forces[member_name][action_name] = {}
                    accumulator = forces[member_name][action_name]

                    # The 'parent_accumulator' and 'path' allow me to update the action
                    # with None if there are no results
                    parent_accumulator = forces[member_name]
                    path = action_name
                else:
                    forces[member_name][action_name][direction] = {}
                    accumulator = forces[member_name][action_name][direction]

                    # The 'parent_accumulator' and 'path' allow me to update the action
                    # with None if there are no results
                    parent_accumulator = forces[member_name][action_name]
                    path = direction
        
                # ...and for each load combo...
                for load_combo in load_combinations:
                    load_combo_name = load_combo['name']
                    accumulator[load_combo_name] = {}
                    
                    # Get the max and min value from the model
                    if direction not in ("axial", "torque"):
                        max_value = float(max_method(direction, load_combo_name))
                        min_value = float(min_method(direction, load_combo_name))
                    else:
                        max_value = float(max_method(load_combo_name))
                        min_value = float(min_method(load_combo_name)) 

                    accumulator[load_combo_name].update({f"max": max_value})
                    accumulator[load_combo_name].update({f"min": min_value})
            if parent_accumulator[path] is None:
                parent_accumulator.pop(path)
    return forces
        

def extract_member_forces_at_locations(
    model: "Pynite.FEModel3D", 
    force_extraction_locations: dict[str, list[float]],
    load_combinations: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Extracts forces at selected members at the locations specified.

    'force_extraction_locations': a dict in the following format:
        {"member01": [0, 2200, 4300], "member02": [3423, 1500]}
        Where:
        - "member01" is a member name
        - The values are a list of locations on the member from which to
            extract results from.

    """
    force_locations = {}

    actions = {
        "shear": ["Fy", "Fz"], # action: [possible direction(s)]
        "moment": ["Mz", "My"],
        "axial": ["axial"], # There are no axial directions; axial is axial!
        "torque": ["torque"], # ...same with torque
    }

    # There many things we need to specify in order to get a number:
    # member_name -> 'forces' -> 'moment'/'shear'/'axial'/'torque' -> Optional[direction] -> load_combo -> 'max'/'min'
    # Which is why there is a loop *four* layers deep
    #
    # This is also complicated by the fact that Pynite subcategorizes forces into
    # "shear", "moment", "axial", and "torque". "shear" and "moment" require an additional
    # direction specification to retrieve the value whereas "axial" and "torque" do not.

    # For each member...
    force_locations = {}
    for member_name, member in model.members.items():
        if member_name not in force_extraction_locations:
            continue
        force_locations[member_name] = {}
        for loc in force_extraction_locations[member_name]:
            force_locations[member_name][loc] = {}
            accumulator = force_locations[member_name][loc]
            for load_combo in load_combinations:
                load_combo_name = load_combo['name']
                accumulator[load_combo_name] = {}
                for action_name, directions in actions.items():
                    # ...and for each direction...
                    for direction in directions:
                        force_method = getattr(member, action_name)
                        if action_name == direction:
                            force_name = action_name
                        else:
                            force_name = f"{action_name}-{direction.lower()}"
                        if direction not in ("axial", "torque"):
                            force_value = float(force_method(direction, loc, load_combo_name))
                        else:
                            force_value = float(force_method(loc, load_combo_name))
        
                        accumulator[load_combo_name].update({force_name: force_value})
    return force_locations


def extract_span_max_mins(
    model: "Pynite.FEModel3D", 
    load_combinations: Optional[list[str]] = None, 
    actions: Optional[list[str]] = None
) -> dict:
    """
    
        {
            "member": {
                "LC": {
                    "M1": {
                        "span_envelope_max": [[Yi, Xi], [Yi, Xi], ...[Yn, Xn]],
                        "span_envelope_max": [
                            {"value": Yi, "loc_abs": Xi, "loc_rel": xi, "span": Li},
                            {"value": Yi, "loc_abs": Xi, "loc_rel": xi, "span": Li},
                        ]
                    },

                }
            }
        }
    """
    if actions is None:
        actions = ['Fy', 'Fz', 'My', 'Mz', 'axial', 'torque', 'dy', 'dx']
    action_methods = {
        "Fy": "shear",
        "Fz": "shear",
        "My": "moment",
        "Mz": "moment",
        "axial": "axial",
        "torque": "torque",
        "dy": "deflection",
        "dz": "deflection",
    }
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
            for action in actions:
                span_envelopes[member_name][lc].setdefault(action, {})
                method_type = action_methods[action]
                method_name = f"{method_type}_array"
                method = getattr(sub_members, method_name)
                if action not in ('axial', 'torque'):
                    result_arrays = method(action, n_points=n_points, combo_name=lc)
                else:
                    result_arrays = method(n_points=n_points, combo_name=lc)
                for envelope in max_min:
                    envelope_key = f"span_envelope_{envelope}"
                    span_envelopes[member_name][lc][action].setdefault(envelope_key, [])
                    length_counter = 0
                    for sub_member in sub_members:
                        locator_func = getattr(np, f'arg{envelope}')
                        envelope_func = getattr(np, envelope)
                        envelope_val = envelope_func(result_arrays[1])
                        envelope_idx = locator_func(result_arrays[1])
                        x_val_local = result_arrays[0][x_val_local]
                        x_val_global = x_val_local + length_counter
                        x_length = result_arrays[0][-1]
                        length_counter += x_length
                        span_envelope = {
                            "value": envelope_val,
                            "loc_rel": x_val_local,
                            "loc_abs": x_val_global,
                            "length": member_length
                        }
                        span_envelopes[member_name][lc][action][envelope_key].append(span_envelope)
    return span_envelopes


        




def extract_spans(model: "Pynite.FEModel3D") -> dict[str, list["Pynite.Member3D"]]:
    """
    Extracts the sub-members for all of the members in the 'model'
    """
    member_spans = {}
    for member_name, member in model.members.items():
        member_spans.setdefault(member_name, [])
        for name, span_member in model.members['My beam'].sub_members.items():
            member_spans[member_name].append(span_member)
    return member_spans


def extract_load_combinations(model: "Pynite.FEModel3D") -> list[str]:
    """
    Returns a list of the load combination names used in the model
    """
    return list(model.load_combos.keys())