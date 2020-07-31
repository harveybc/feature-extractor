import os
import json
from flask import Flask
from feature_extractor.feature_extractor import FeatureExtractor

def read_plugin_config(vis_config_file=None):
    """ Read the pulgin configuration JSON file from a path, if its None, uses a default configuration """
    if vis_config_file != None:
        file_path = vis_config_file
    else:
        file_path = os.path.dirname(os.path.abspath(__file__)) + "//visualizer.json"
    with open(file_path) as f:
        data = json.load(f)
    return data
	
def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # read plugin configuration JSON file
    p_config = read_plugin_config()
    # initialize FeatureExtractor
    fe = FeatureExtractor(p_config)
    # set flask app parameters
    app.config.from_mapping(
        # a default secret that should be overridden by instance config
        SECRET_KEY="dev",
        # store the database in the instance folder
        DATABASE=os.path.join(BASE_DIR, "test.sqlite"),
        # plugin configuration from visualizer.json
        P_CONFIG = p_config, 
        # feature_extractor instance with plugins already loaded
        FE = fe
    )
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.update(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    #@app.route("/hello")
    #def hello():
    #    return "Hello, World!"

    # register the database commands
    from feature_extractor.visualizer import db

    db.init_app(app)

    
    # TODO: replace 0 in vis_data by process_id, obtained as the first process_id belonging to the current user.  
    # read the data to be visualized using the using the Feature extractor instance, preinitialized in __init__.py with input and output plugins entry points.
    vis_data = app.config['FE'].ep_input.load_data(app.config['P_CONFIG'], 0)
    # use the output plugin entry point to get the path of the template for the visualizer blueprint
    plugin_folder = app.config['FE'].ep_output.template_path(app.config['P_CONFIG'])

    # construct the visualizer blueprint using the plugin folder as template folder
    bp = Blueprint("visualizer", __name__,  template_folder=app.config['FE'].ep_output.template_path(app.config['P_CONFIG']))
    
    # apply the blueprints to the app
    from feature_extractor.visualizer import auth, visualizer






    app.register_blueprint(auth.bp)
    app.register_blueprint(visualizer.bp) 

    # make url_for('index') == url_for('blog.index')
    # in another app, you might define a separate main index here with
    # app.route, while giving the blog blueprint a url_prefix, but for
    # the tutorial the blog will be the main index
    app.add_url_rule("/", endpoint="index")

    return app
