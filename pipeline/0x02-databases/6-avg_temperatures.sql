-- calculate the average of score in temperatures table
-- database name come in Command  line
SELECT city, AVG(value) AS avg_tmp FROM temperatures GROUP BY city ORDER BY avg_tmp DESC;
