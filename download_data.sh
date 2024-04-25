mkdir data
wget http://linghub.ru/static/Taiga/news.zip
unzip news.zip 'Lenta/*' -d data
rm news.zip
