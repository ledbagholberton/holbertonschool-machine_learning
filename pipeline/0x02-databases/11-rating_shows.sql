-- lists all shows from hbtn_0d_tvshows_rate by their rating
-- use of inner join i three levelss only one SELEcT
SELECT tv_shows.title, SUM(rate) AS rating
FROM tv_show_ratings
INNER JOIN tv_shows
ON tv_shows.id = tv_show_ratings.show_id
GROUP BY show_id ORDER BY rating DESC;
