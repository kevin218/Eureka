from . import Model

class CompositeModel(Model):
    """A class to create composite models"""
    def __init__(self, models, **kwargs):
        """Initialize the composite model

        Parameters
        ----------
        models: sequence
            The list of models
        """
        # Inherit from Model calss
        super().__init__(**kwargs)

        # Store the models
        self.components = models

    def eval(self, **kwargs):
        """Evaluate the model components"""
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Empty flux
        flux = 1.

        # Evaluate flux at each model
        for model in self.components:
            if model.time is None:
                model.time = self.time
            flux *= model.eval(**kwargs)

        return flux

    def syseval(self, **kwargs):
        """Evaluate the systematic model components only"""
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Empty flux
        flux = 1.

        # Evaluate flux at each model
        for model in self.components:
            if model.modeltype == 'systematic':
                if model.time is None:
                    model.time = self.time
                flux *= model.eval(**kwargs)

        return flux

    def physeval(self, **kwargs):
        """Evaluate the physical model components only"""
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Empty flux
        flux = 1.

        # Evaluate flux at each model
        for model in self.components:
            if model.modeltype == 'physical':
                if model.time is None:
                    model.time = self.time
                flux *= model.eval(**kwargs)

        return flux

    def update(self, newparams, names, **kwargs):
        """Update parameters in the model components"""
        # Evaluate flux at each model
        for model in self.components:
            model.update(newparams, names, **kwargs)

        return
