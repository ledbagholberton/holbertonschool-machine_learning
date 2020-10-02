-- lists all genres in the database hbtn_0d_tvshows_rate by their rating
-- use of inner join i three levelss only one SELEcT
SELECT tv_genres.name, SUM(rate) AS rating
FROM tv_show_ratings
INNER JOIN tv_shows
ON tv_shows.id = tv_show_ratings.show_id
INNER JOIN tv_show_genres
ON tv_shows.id = tv_show_genres.genre_id
INNER JOIN tv_genres
ON tv_show_genres.genre_id = tv_genres.id
GROUP BY tv_genres.id ORDER BY rating DESC;
