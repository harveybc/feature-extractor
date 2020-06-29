-- Initialize the database.
-- Drop any existing data and create empty tables.

DROP TABLE IF EXISTS processes;
DROP TABLE IF EXISTS training_progress;
DROP TABLE IF EXISTS validation_plots;
DROP TABLE IF EXISTS validation_stats;

CREATE TABLE "processes" (
	"id"	INTEGER,
	"name"	TEXT,
	"description"	TEXT,
	"model_link"	TEXT,
	"training_data_link"	TEXT,
	"validation_data_link"	TEXT,
	PRIMARY KEY("id" AUTOINCREMENT)
)

CREATE TABLE "training_progress" (
	"id"	INTEGER,
	"mse"	DOUBLE,
	"mae"	DOUBLE,
	"r2"	DOUBLE,
	"created"	INTEGER  NOT NULL DEFAULT CURRENT_TIMESTAMP,
	"process_id"	INTEGER,
	PRIMARY KEY("id" AUTOINCREMENT)
    FOREIGN KEY (process_id) REFERENCES processes (id)
)

CREATE TABLE "validation_plots" (
	"id"	INTEGER,
	"original"	DOUBLE,
	"predicted"	DOUBLE,
	"predicted2"	DOUBLE,
	"created"	INTEGER NOT NULL DEFAULT CURRENT_TIMESTAMP,
	"process_id"	INTEGER,
	PRIMARY KEY("id" AUTOINCREMENT)
    FOREIGN KEY (process_id) REFERENCES processes (id)
)

CREATE TABLE "validation_stats" (
	"id"	INTEGER,
	"mse"	DOUBLE,
	"mae"	DOUBLE,
	"r2"	DOUBLE,
	"created"	INTEGER NOT NULL DEFAULT CURRENT_TIMESTAMP,
	"process_id"	INTEGER,
	PRIMARY KEY("id" AUTOINCREMENT)
    FOREIGN KEY (process_id) REFERENCES processes (id)
)
