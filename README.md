Код для обучения word2vec модели на русском с нуля, используя pytorch.
В качестве датасета используется новостной срез датасета Taiga.
# Загрузка данных
Сначала запустить скрипт download_data, он скачает данные и разархивирует данные в папку data (нужен wget для запуска). 
```
./download_data.sh
```
В данном коде будем использовать данные только из папки Lenta.
Все файлы в Lenta/texts содержат одну пустую строку и затем строку с предложениями.
Скрипт `merge_files.rs` собирает все предложения по всем файлам и записывает их, каждое в одной строке, в новый файл `sentences.txt`.
Написан на rust ибо я даже не хочу проверять сколько времени уйдёт на это у python.

Компилируем скрипт:
```
rustc merge_files.rs
```
И запускаем его `./merge_files`.

# TODO
* Натренировать модель и сохранить её
* Возможно добавить визуализацию векторов слов для оценки работу модели
