# Easily extract results from Pynite... with `pynite_reporting`!

PyniteFEA is excellent and it is generally design-ready...if it weren't for all the trouble we have to go through to get results out. This is not unique to Pynite, most FEA programs require a significant amount of post-processing to prepare the actual analysis results for design.

## Enter `pynite_reporting`

This package provides a series of functions that consume a `Pynite.FEModel3D` object and returns consistently-structured dictionaries of analysis results.

> **Note:** As of 2025-06-19, this package has only been "casually tested" (meaning simple visual checking of outputs). No test suite has been written (but is coming).


## Installation

```
pip install pynite_reporting
```

## Dependencies

- Numpy (>= 2.0.0)

(PyNiteFEA is not a dependency but it is assumed to be in your working environment)

### Pynite Compatibility

`PyniteFEA >= 1.0.0`

(Not compatible with pre-v1.0 versions!)


## Examples (typical use)

```python
from Pynite import FEModel3D
import pynite_reporting as pr

model = FEModel3D(...) # Build your model here

# Selected load combinations in your model
lcs = [
    # 'LC1', 
    'LC2',
    'LC3',
    # 'LC4', 
    # 'LC5',
]

# All the below functions optionally take a list of load combos
# so you can select which combos to extract

# Return reactions for all supports, all load combos
reactions = pr.extract_reactions(
    model,
    # load_combinations=lcs
)

# Return force arrays for all members, all load combos
force_arrays = pr.extract_member_force_arrays(
    model,
    # load_combinations=lcs
)

# Return force min/max envelope for all members, all load combos
# Min/Max values will not necessarily be at concurrent locations
forces_minmax = pr.extract_member_forces_minmax(
    model,
    # load_combinations=lcs
)

# Return force min/max envelope for each span in all members, all load combos
forces_minmax_spans = pr.extract_span_forces_minmax(
    model,
    # load_combinations=lcs
)

# Return forces for all load combos at specific locations along the global member length
forces_at_locations = pr.extract_member_forces_at_locations(
    model, 
    force_extraction_locations={"Member01": [0, 2000, 3600]},
    # load_combinations=lcs
)

# Return forces for all load combos at 1/4 points for each span of the given members
forces_at_location_ratios = pr.extract_member_forces_at_locations(
    model, 
    force_extraction_ratios={"Member05": [0.25, 0.5, 0.75]}, 
    by_span=True,
    # load_combinations=lcs
    )

# Returns all node deflections for all load combos
node_deflections = pr.extract_node_deflections(
    model,
    # load_combinations=lcs
)
```

**And there you have it!** Does that not make your life a little bit easier?

## FYI (Opinions at work!)

I have made the decision to _remove unnecessary results_ from being returned by _some_ of these functions.

**"???WHAA??? I want to see ALL of my results!!!"**, you say?

I don't think you actually do. Consider the following _small amount_ of results:

```python
{
    'M_col': {
        'shear': {
            'Fy': {
                'LC1': {'max': 0, 'min': 0}, # No loading for this load case on this member
                'LC2': {'max': 4000, 'min': 4000}
            },
            'Fz': {
                'LC1': {'max': 10000, 'min': 10000}, 
                'LC2': {'max': 0, 'min': 0} # No loading for this either...
            },
            
        },
        'moment': {
            'Mz': {
                'LC1': {'max': 0, 'min': 0}, # Or this...
                'LC2': {'max': 20000, 'min': 0}
            }, 
            'My': {
                'LC1': {'max': 50000, 'min': 50000},
                'LC2': {'max': 0, 'min': 0} # Or this...
            }
        },
        ...
    }
}
```

The above results contain _unnecessary data_. This structure has loading in the gravity direction and the transverse direction, each on a different load case/combo.

The load cases that show as `0`, `0` indicate that the force diagrams are completely flat and without activity.

To avoid confusion in reading and to prevent unnecessary iterations (if you are putting these results through an automated process), I have filtered out the keys that result in null values.

Here is how the above results are returned:

```python
    'M_col': {
        'shear': {
            'Fy': {
                'LC2': {'max': 4000, 'min': 4000}
            },
            'Fz': {
                'LC1': {'max': 10000, 'min': 10000}, 
            },
        },
        'moment': {
            'Mz': {
                'LC2': {'max': 20000, 'min': 0}
            }, 
            'My': {
                'LC1': {'max': 50000, 'min': 50000}
            }
        },
        ...
    },
```

So, you know that all other load combos result in null values _without having to physically read a bunch of zeros or confusing "near zero" values._

The tolerance for this is an absolute tolerance of `1e-8`. Currently, this is not parameterized and is hard-coded into the package (because it was easier and made the function signatures cleaner). So, even if you have REALLY small result values (on the order of `0.0000001` units), those values will still be returned to you (and not excluded).

**Note: Not ALL functions have this behaviour**

Functions which will always return all results:

* `pynite_reporting.extract_member_forces_at_locations`

This is allows you to see all concurrent forces for a load combination at a given location.




