# cython: language_level=3
"""DVM pass manager -- phase-1 no-op implementation.

In phase 1, no optimization passes are applied.  The pass manager simply
returns the input object list unchanged.
"""

cpdef list run_passes(list objects):
    """Run all registered passes over the object list.

    Phase 1: returns the input list unchanged (no-op).

    Parameters
    ----------
    objects : list
        List of NDObject instances forming the computation graph.

    Returns
    -------
    list
        The same list, unmodified.
    """
    return objects
