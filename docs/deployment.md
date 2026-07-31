Документация по развертыванию

Для оптимизации инференса модель можно конвертировать в формат ONNX:

import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('float_input', FloatTensorType([None, 23]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("models/model_v1.onnx", "wb") as f:
f.write(onnx_model.SerializeToString())

Преимущества ONNX:
- ускорение предсказаний за счет ONNX Runtime
- независимость от Python-окружения (можно деплоить на C++/Java серверах)
- оптимизация графа вычислений (константный фолдинг, удаление лишних слоев)

Встроенный сервер Flask (app.run()) не предназначен для production. В реальной среде используется связка uWSGI + NGINX:

uWSGI:
- WSGI-сервер, который запускает Python-приложение в нескольких воркерах
- обрабатывает запросы параллельно, перезапускает упавшие процессы
- пример запуска: uwsgi --http 0.0.0.0:5000 --wsgi-file app/app.py --callable app --processes 4

NGINX:
- reverse proxy и веб-сервер
- принимает внешние запросы (порты 80/443), отдает статику
- проксирует динамические запросы на uWSGI через unix-socket или HTTP
- балансирует нагрузку и защищает от DDoS

Схема работы:

Client -> NGINX (80/443) -> uWSGI (socket/HTTP) -> Flask App

Такая архитектура обеспечивает отказоустойчивость и масштабируемость.