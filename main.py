import os
import base64
import fitz
from openai import OpenAI
from PIL import Image
from io import BytesIO

# ==================================================
# НАСТРОЙКИ
# ==================================================

PDF_FILE = "file.pdf"

OPENROUTER_API_KEY = "API_KEY"

MODELS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3.5-super-vl:free"
]

START_PAGE = 1
END_PAGE = 999999

# Количество страниц в одном запросе
PAGES_PER_REQUEST = 10

IMAGE_DPI = 150

# ==================================================
# OPENROUTER
# ==================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    max_retries=0
)


def page_to_image(page):

    pix = page.get_pixmap(
        matrix=fitz.Matrix(IMAGE_DPI / 72, IMAGE_DPI / 72),
        alpha=False
    )

    image = Image.open(BytesIO(pix.tobytes("png")))
    image.thumbnail((2000, 2000))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def recognize_batch(images, start, end):

    prompt = f"""
Ты — профессиональная система OCR.

Перед тобой {len(images)} страниц PDF.

Распознай ВСЁ содержимое всех страниц.

ОБЯЗАТЕЛЬНО:

1. Извлекай весь текст без сокращений.
2. Сохраняй заголовки и абзацы.
3. Распознавай текст на фотографиях и изображениях.
4. Таблицы преобразуй в текстовые таблицы с разделителем |.
5. Сохраняй числа, даты, формулы и единицы измерения.
6. Описывай графики, схемы и диаграммы.
7. Сохраняй порядок страниц.
8. Не придумывай отсутствующий текст.
9. Нечитаемые места отмечай [НЕРАЗБОРЧИВО].
10. Не пропускай страницы.
11. Не пересказывай и не сокращай содержимое.

Каждая страница должна начинаться с заголовка:

СТРАНИЦА N

Обработай страницы с {start} по {end}.

Верни только распознанный текст.
"""

    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    for index, image in enumerate(images):

        content.append({
            "type": "text",
            "text": f"Изображение страницы {start + index}:"
        })

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image}"
            }
        })

    last_error = None

    for model in MODELS:

        try:

            print(f"Используется модель: {model}")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=0.1,
                max_tokens=32000
            )

            result = response.choices[0].message.content

            if result and result.strip():
                return result.strip()

            last_error = "Пустой ответ модели"

        except Exception as error:

            last_error = str(error)
            print(f"Ошибка модели: {last_error}")

    return f"[ОШИБКА: {last_error}]"


# ==================================================
# ОСНОВНАЯ ПРОГРАММА
# ==================================================

def main():

    if not os.path.exists(PDF_FILE):
        print("PDF-файл не найден!")
        return

    document = fitz.open(PDF_FILE)

    total_pages = len(document)

    start = max(1, START_PAGE)
    end = min(END_PAGE, total_pages)

    if start > end:
        print("Неверный диапазон страниц!")
        document.close()
        return

    output_file = os.path.splitext(PDF_FILE)[0] + ".txt"

    total_to_process = end - start + 1

    total_batches = (
        total_to_process + PAGES_PER_REQUEST - 1
    ) // PAGES_PER_REQUEST

    print("=" * 50)
    print(f"PDF: {PDF_FILE}")
    print(f"Всего страниц: {total_pages}")
    print(f"Обрабатывается: {total_to_process}")
    print(f"Страниц за запрос: {PAGES_PER_REQUEST}")
    print(f"Всего запросов: {total_batches}")
    print("=" * 50)

    with open(output_file, "w", encoding="utf-8") as output:

        batch_number = 0

        for batch_start in range(start, end + 1, PAGES_PER_REQUEST):

            batch_number += 1

            batch_end = min(
                batch_start + PAGES_PER_REQUEST - 1,
                end
            )

            print(
                f"\nПакет {batch_number}/{total_batches}: "
                f"страницы {batch_start}-{batch_end}"
            )

            images = []

            # Подготавливаем страницы
            for page_number in range(batch_start, batch_end + 1):

                print(f"Подготовка страницы {page_number}")

                page = document[page_number - 1]
                image = page_to_image(page)

                images.append(image)

            # Один запрос на весь пакет
            print("Отправка пакета в OpenRouter...")

            result = recognize_batch(
                images,
                batch_start,
                batch_end
            )

            # Сохраняем результат
            output.write(
                f"\n\n{'=' * 50}\n"
                f"ПАКЕТ {batch_number}: "
                f"СТРАНИЦЫ {batch_start}-{batch_end}\n"
                f"{'=' * 50}\n\n"
            )

            output.write(result)
            output.flush()

            print(f"Пакет {batch_number} завершён!")

    document.close()

    print("\n" + "=" * 50)
    print("ОБРАБОТКА ЗАВЕРШЕНА!")
    print(f"Результат: {output_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
