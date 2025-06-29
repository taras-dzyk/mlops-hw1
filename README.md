# Лабораторні роботи MLOps
Цей репозиторій містить всі інструменти і налаштування оточення, що я використав для виконання лабораторних робіт.

В реальному проекті версіювання було би організовано окремим репозиторієм.

### Зміст

1. Лабораторна робота 1: Робота з даними в MLOps
    - [розмітка даних](#як-запуститивідкрити-розмітку-даних)
    - [версіювання датасету](#як-працює-версіонування-датасету)
2. Лабораторна робота 2: Тренування моделей та трекінг експериментів
    - [тренування моделі](#як-тренувати-модель)
    - [трекінг експериментів та сховище моделей](#трекінг-експериментів-та-сховище-моделей)
3. Лабораторна робота 3: Інференс моделей
    - [завантаження і розгортання моделі](#інференс-моделей)
4. Лабораторна робота 4: Моніторинг та Observability
    - [логування базових метрик](#логування-базових-метрик)
4. [Загальна структура репозиторію](#загальна-структура-репозиторію)
5. [Подальші кроки](#подальші-кроки)

### Як запустити/відкрити розмітку даних
Розмітку даних виконано в [Label Studio](https://labelstud.io/)

**Передумови:**
- встановлений docker compose

**Кроки:**

перебуваючи в коренні репозиторію 

1. `cd labeling/label-studio/`

2. `docker-compose up -d`  

    Це запустить Label Studio на http://localhost:8080/

    (налаштування порту є в `docker-compose.yml`)

3. Log in:  
    ```
    t.dzyk@setuniversity.edu.ua
    JPv$u#DaXP4m4Pr
    ```
4. Recent Projects -> MLOps-hw-1

### Як працює версіонування датасету

Для версіювання наборів даних я використав [DVC](https://dvc.org/)


**Передумови:**
- встановлений DVC

перебуваючи в коренні репозиторію 

`cd versioning/dvc`

Це основний репозиторій версіювання. 

Я ініціалізував репозиторій в режимі `dvc init --subdir` щоб зробити його розміщення можливим не в корені git репозиторію. Це було зроблено для цієї лабораторної.

Я додав папку `dataset` в dvc repo, куди я поміщаю розмічені набори даних. Ця папка додалася в `.gitignore` на цьому ж рівні, що виключило власне сам датасет, який може бути дуже великим з git трекінгу.

Я налаштував dvc data storage для цієї лабораторної на локальному середовищі. `../dvc_storage/`. Це можна побачити в файлі `.dvc/config`

Відповідно при виконанні `dvc push` дані будуть направлені в сховище. В реальному проекті це буде клауд бакет

### Як тренувати модель

Перебуваючи в корені репозиторію

`cd training/ray`

`docker-compose up --build -d`

Це підніме локальний кластер [Ray](https://docs.ray.io/en/latest/train/train.html)

Дешборд буде доступний: http://localhost:8265

Тестовий запуск можна виконати

`docker exec -it ray-head bash`

`python test_ray.py`

Паралельне тренування тестових tensorflow моделей з різними параметрами

`docker exec -it ray-head bash`

`python train_img_ray.py`

### Трекінг експериментів та сховище моделей
Як трекер експериментів та сховище моделей я використав [W&B](https://wandb.ai/)

У мене є акаунт на цьому сервісі і він інтенрований в тренування моделі через 

`WANDB_API_KEY` Апі ключ

`WANDB_ENTITY` Ідентифікатор команди(щось типу неймспейсу для групування) в W&B

Вони обидва передаються як змінні оточення через `docker-compose.yml`

### Інференс моделей
Cховище моделей: [W&B](https://wandb.ai/)
Локальний інференс на кластері [Ray](https://docs.ray.io)

`WANDB_API_KEY` Апі ключ

`WANDB_ENTITY` Ідентифікатор команди(щось типу неймспейсу для групування) в W&B

Вони обидва передаються як змінні оточення через `docker-compose.yml`

`docker exec -it ray-head bash`

`python run_img_ray.py`

Зображення розмістити в папку `input`. Скрипт пробігається по ним усім і ті, що ідентифіковані як крокодили копіює в папку `output`

### Логування базових метрик
- Кількість передбачень за хвилину

- Середній час обробки

Я використовую поєднання [Prometheus](https://prometheus.io/) для метрик і [Grafana](https://grafana.com/) для візуалізації дашбордів

наразі логується 

`predictions_total`

`predictions_positive_total`

`prediction_duration_seconds`


### Загальна структура репозиторію
- labeling
    - image_library
    
            Бібліотека зображень

    - label-studio

            Містить знімок мого локального Label Studio розгорнутого через docker compose

- versioning
    - dvc

            Власне dvc репозиторій, де версіюються набори даних. В реальному проекті саме це буде основним репозиторієм версіювання
    - dvc_storage
        
            Локальний data storage для dvc
- training
    - ray
        - ray-head
            
                Head нода ray
        - ray-worker
                
                Worker нода ray
- input
        
        тут розміщуються вхідні тестові зображення

- output
        
        сюди програма копіює позитивно класифіковані зображення

- models

        претренована модель

- dataset

        тренувальний датасет

### Подальші кроки

Я планую додати логування і пізніше розділити на репозиторій для версіювання і решту

