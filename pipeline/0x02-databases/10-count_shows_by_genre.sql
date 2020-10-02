-- from exissting db show join of three tables
-- count by genre
SELECT tv_genres.name, COUNT(tv_genres.name) 
AS number_of_shows 
FROM tv_genres 
INNER JOIN tv_show_genres 
ON tv_genres.id = tv_show_genres.genre_id 
GROUP BY tv_show_genres.genre_id 
ORDER BY number_of_shows DESC;
