
# This file contains the visualizer plugin, th input plugin can load all the data or starting from
 # the last id.

from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from werkzeug.exceptions import abort

from feature_extractor.visualizer.auth import login_required
from feature_extractor.visualizer.db import get_db
from flask import current_app


def visualizer_blueprint(plugin_folder):

    # construct the visualizer blueprint using the plugin folder as template folder
    bp = Blueprint("visualizer", __name__,  template_folder=plugin_folder)
    
    @bp.route("/")
    @login_required
    def index():
        # read the data to be visualized using the using the Feature extractor instance, preinitialized in __init__.py with input and output plugins entry points.
        # TODO: replace 0 in vis_data by process_id, obtained as the first process_id belonging to the current user.    
        vis_data = current_app.config['FE'].ep_input.load_data(current_app.config['P_CONFIG'], 0)
        return render_template("/plugin_templates/visualizer/index.html", p_config = current_app.config['P_CONFIG'], vis_data =  vis_data)


    def get_post(id, check_author=True):
        """Get a post and its author by id.

        Checks that the id exists and optionally that the current user is
        the author.

        :param id: id of post to get
        :param check_author: require the current user to be the author
        :return: the post with author information
        :raise 404: if a post with the given id doesn't exist
        :raise 403: if the current user isn't the author
        """
        results = (
            get_db()
            .execute(
                "SELECT p.id, title, body, created, author_id, username"
                " FROM post p JOIN user u ON p.author_id = u.id"
                " WHERE p.id = ?",
                (id,),
            )
            .fetchone()
        )
        # verify if the query returned no results
        if results is None:
            abort(404, "Post id {id} doesn't exist.")
            
        return results


    @bp.route("/create", methods=("GET", "POST"))
    @login_required
    def create():
        """Create a new post for the current user."""
        if request.method == "POST":
            title = request.form["title"]
            body = request.form["body"]
            error = None

            if not title:
                error = "Title is required."

            if error is not None:
                flash(error)
            else:
                db = get_db()
                db.execute(
                    "INSERT INTO post (title, body, author_id) VALUES (?, ?, ?)",
                    (title, body, g.user["id"]),
                )
                db.commit()
                return redirect(url_for("visualizer.index"))

        return render_template("visualizer/create.html")


    @bp.route("/<int:id>/update", methods=("GET", "POST"))
    @login_required
    def update(id):
        """Update a post if the current user is the author."""
        post = get_post(id)

        if request.method == "POST":
            title = request.form["title"]
            body = request.form["body"]
            error = None

            if not title:
                error = "Title is required."

            if error is not None:
                flash(error)
            else:
                db = get_db()
                db.execute(
                    "UPDATE post SET title = ?, body = ? WHERE id = ?", (title, body, id)
                )
                db.commit()
                return redirect(url_for("visualizer.index"))

        return render_template("visualizer/update.html", post=post)


    @bp.route("/<int:id>/delete", methods=("POST",))
    @login_required
    def delete(id):
        """Delete a post.

        Ensures that the post exists and that the logged in user is the
        author of the post.
        """
        get_post(id)
        db = get_db()
        db.execute("DELETE FROM post WHERE id = ?", (id,))
        db.commit()
        return redirect(url_for("visualizer.index"))
    
    return bp
