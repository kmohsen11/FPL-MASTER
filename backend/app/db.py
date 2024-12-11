import sqlite3
import click
from flask import current_app, g
from flask_sqlalchemy import SQLAlchemy

# Flask-SQLAlchemy instance
db = SQLAlchemy()

# SQLite connection helpers (rename to avoid conflict)
def get_sqlite_db():
    if 'sqlite_db' not in g:
        g.sqlite_db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.sqlite_db.row_factory = sqlite3.Row
    return g.sqlite_db

def close_sqlite_db(e=None):
    sqlite_db = g.pop('sqlite_db', None)

    if sqlite_db is not None:
        sqlite_db.close()

def init_sqlite_db():
    sqlite_db = get_sqlite_db()
    with current_app.open_resource('schema.sql') as f:
        sqlite_db.executescript(f.read().decode('utf8'))

@click.command('init-sqlite-db')
def init_sqlite_db_command():
    init_sqlite_db()
    click.echo('Initialized the SQLite database.')

def init_app(app):
    app.teardown_appcontext(close_sqlite_db)
    app.cli.add_command(init_sqlite_db_command)
