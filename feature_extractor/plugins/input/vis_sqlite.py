# -*- coding: utf-8 -*-
"""
This File contains the LoadCSV class plugin. 
"""

from feature_extractor.plugin_base import PluginBase
from numpy import genfromtxt
from sys import exit

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class VisSqlite(PluginBase): 
    """ input plugin for the FeatureExtractor class, after initialization, the input_ds attribute is set """
    
    def __init__(self, conf):
        """ Initializes PluginBase. Do NOT delete the following line whether you have initialization code or not. """
        super().__init__(conf)
        # Insert your plugin initialization code here.
        pass

    def parse_cmd(self, parser):
        """ Adds command-line arguments to be parsed, overrides base class """
        parser.add_argument("--input_file", help="Input dataset file to load including path.", required=True)
        return parser
    
    def load_data(self):
        """ Load the input dataset """
        self.input_ds = genfromtxt(self.conf.input_file, delimiter=",")


    """Show the mse plot for the last training process, also the last validation plot and a list of validation stats."""
    p_config = current_app.config['P_CONFIG']
    db = get_db()
    training_progress = db.execute(
        "SELECT *"
        " FROM training_progress t JOIN process p ON t.process_id = p.id"
        " ORDER BY created DESC"
    ).fetchall()
    validation_plots = db.execute(
        "SELECT *"
        " FROM validation_plots t JOIN process p ON t.process_id = p.id"
        " ORDER BY created DESC"
    ).fetchall()
    validation_stats = db.execute(
        "SELECT *"
        " FROM validation_stats t JOIN process p ON t.process_id = p.id"
        " ORDER BY created DESC"
    ).fetchall()



        return self.input_ds
        
