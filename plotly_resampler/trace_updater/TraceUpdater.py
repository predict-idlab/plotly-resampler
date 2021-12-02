# AUTO GENERATED FILE - DO NOT EDIT

from dash.development.base_component import Component, _explicitize_args


class TraceUpdater(Component):
    """A TraceUpdater component.
TraceUpdater is a component which updates the trace of a given figure.
It takes the properties
 - gdID - which is the DCC.graph its id
 - updatedData - A list whose
    * first object withholds the to-be applied layout
    * second to ... object contain the updated trace data and its corresponding
      index under the `index` attribute

Keyword arguments:

- id (string; optional):
    The ID used to identify this component in Dash callbacks.

- gdID (string; required):
    The id of the graph-div whose traces should be.

- updateData (list; optional):
    The data to update the graph with, must contain the `index`
    property for each trace; either a list of dict-traces or a single
    trace."""
    @_explicitize_args
    def __init__(self, id=Component.UNDEFINED, gdID=Component.REQUIRED, updateData=Component.UNDEFINED, **kwargs):
        self._prop_names = ['id', 'gdID', 'updateData']
        self._type = 'TraceUpdater'
        self._namespace = 'trace_updater'
        self._valid_wildcard_attributes =            []
        self.available_properties = ['id', 'gdID', 'updateData']
        self.available_wildcard_properties =            []
        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs
        args = {k: _locals[k] for k in _explicit_args if k != 'children'}
        for k in ['gdID']:
            if k not in args:
                raise TypeError(
                    'Required argument `' + k + '` was not specified.')
        super(TraceUpdater, self).__init__(**args)
