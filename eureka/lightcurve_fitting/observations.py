"""Base class to handle observation metadata

Author: Laura Kreidberg 
Email: kreidberg@mpia.de
"""



class Observations(object):
    """
    A class to store meta-data related to the observation

    :param nvisit: number of visits in the observation.
    :type nvisit: int
    
    :param nexposure: number of total exposures over all visits.
    :type nexposure: int
    
    """
    def __init__(self):
        # Set the attributes
        self.nvisit = None
        self.nexposure = None

