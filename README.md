# Автоматизированная система переименования и классификации изображений на основе их содержания с использованием глубоких нейронных сетей

![Python](https://img.shields.io/badge/Python-3.9.0-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0_CUDA12.1-red)
![Gradio](https://img.shields.io/badge/Gradio-5.0.1-orange)

Система для автоматического анализа изображений с возможностью:
- Генерации описаний с помощью ИИ-моделей
- Классификации изображений по содержанию
- Переименования файлов на основе контента
- Перевода результатов на 50+ языков

## 🌟 Ключевые особенности
- **Поддержка нескольких моделей**:
  - Сегментация: Florence-2 и др.
  - Генерация подписей: BLIP, GIT и др.
  - Перевод: mBART и потенциально др.
- **Гибкий интерфейс**:
  - Редактирование результатов перед сохранением
  - Просмотр оригинальных и обработанных названий изображений
- **Адаптивная обработка**:
  - Автоматическое определение GPU/CPU
  - Поддержка CUDA 12.1 и MPS (Metal)
- **Экспорт данных**:
  - Сохранение в иерархические папки
  - Резервное копирование оригиналов

## 🛠 Установка

### Требования
- Python 3.9–3.11
- NVIDIA GPU (рекомендуется) + [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-downloads)
- 4GB+ оперативной памяти

### Инструкция:
1. Установите [Python 3.9–3.11](https://www.python.org/downloads/) для выполнения скриптов.
2. Установите [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-downloads) для GPU-ускорения
3. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/morzellen/renamer-classifier-images-using-ai.git
   ```
4. Запустите скрипт настройки:
   ```bash
   setup_and_run.bat
   ```

Система автоматически:
- Создаст виртуальное окружение
- Установит зависимости
- Запустит веб-интерфейс

## 🖥 Использование
1. **Загрузка изображений**:
   - Перетащите файлы в область "Загрузите фото"
   - Поддерживаются JPG/PNG/WebP/BMP и множество др.

2. **Настройка моделей**:
   - Выбор моделей через выпадающие списки в интерфейсе
   - Выбор целевого языка перевода

3. **Обработка**:
   - Автоматическая классификация по объектам
   - Генерация описаний на выбранном языке
   - Интерактивное редактирование результатов

4. **Экспорт**:
   - Автоматическое резервное копирование в структурированные папки:
     - `/results/classified_photos` - классифицированные изображения (в подпапках)
     - `/results/renamed_photos` - обработанные названия файлов

## 🗂 Структура проекта
```
.
├── src/                     # Исходный код
│   ├── core/                # Ядро системы
│   │   ├── creators/        # Фабрики моделей
│   │   ├── handlers/        # Логика workflow
│   │   ├── constants/       # Конфигурации
│   │   ├── utils/           # Вспомогательные утилиты
│   │   └── ui/              # Графический интерфейс
│   └── main.py              # Точка входа
├── logs/                    # Логи выполнения
├── venv/                    # Виртуальное окружение
├── installed_libs.txt       # Список зависимостей
└── setup_and_run.bat        # Скрипт установки
```

## ⚙️ Технические детали
- **Мультиязычность**:
  ```python
  # Поддерживаемые языки
  # src\core\constants\web.py
  TRANSLATION_LANGUAGES = {
    "Afrikaans": "af_ZA", "Arabic": "ar_AR",
    "Azerbaijani": "az_AZ", "Bengali": "bn_IN",
    "Burmese": "my_MM", "Chinese": "zh_CN",
    "Croatian": "hr_HR", "Czech": "cs_CZ",
    "Dutch": "nl_XX", "English": "en_XX",
    "Estonian": "et_EE", "Finnish": "fi_FI",
    "French": "fr_XX", "Galician": "gl_ES",
    "Georgian": "ka_GE", "German": "de_DE",
    "Gujarati": "gu_IN", "Hebrew": "he_IL",
    "Hindi": "hi_IN", "Indonesian": "id_ID",
    "Italian": "it_IT", "Japanese": "ja_XX",
    "Kazakh": "kk_KZ", "Khmer": "km_KH",
    "Korean": "ko_KR", "Latvian": "lv_LV",
    "Lithuanian": "lt_LT", "Macedonian": "mk_MK",
    "Malayalam": "ml_IN", "Marathi": "mr_IN",
    "Mongolian": "mn_MN", "Nepali": "ne_NP",
    "Pashto": "ps_AF", "Persian": "fa_IR",
    "Polish": "pl_PL", "Portuguese": "pt_XX",
    "Romanian": "ro_RO", "Russian": "ru_RU",
    "Sinhala": "si_LK", "Slovene": "sl_SI",
    "Spanish": "es_XX", "Swahili": "sw_KE",
    "Swedish": "sv_SE", "Tagalog": "tl_XX",
    "Tamil": "ta_IN", "Telugu": "te_IN",
    "Thai": "th_TH", "Turkish": "tr_TR",
    "Ukrainian": "uk_UA", "Urdu": "ur_PK",
    "Vietnamese": "vi_VN", "Xhosa": "xh_ZA"
  }
  ```
- **Редактирование результатов**:
  - Прямое изменение в таблице интерфейса
  
- **Оптимизации**:
  - Пакетная обработка
  - Кэширование промежуточных результатов
  - Автоматическая очистка памяти

## Лицензия
MIT License. Подробности в файле [LICENSE](LICENSE).

---

**Поддержка**: Для вопросов и предложений создавайте Issues в [репозитории](https://github.com/morzellen/renamer-classifier-images-using-ai.git).  
**Рекомендации**: Для обработки >1000 изображений используйте GPU с 8GB+ видеопамяти.
