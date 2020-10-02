-- show max temperature of each state (ordered by State name)
-- database name come in Command  line
SELECT state, MAX(value) AS max_temp FROM temperatures GROUP BY state ORDER BY state ASC;
