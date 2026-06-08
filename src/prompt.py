import json
import os
from src.utils import ms
from src.analyzer import navec, get_proximity_index_neighbors

# --- ФУНКЦИИ ДЛЯ ПОДГОТОВКИ ЛЛМ-ИНТЕРПРЕТАЦИИ ---

def synonyms_proximity_index(target_word, synonyms_filtered, proximity_weights):
    """
    Создает словарь для LLM, который показывает вес связи каждого синонима с таргетом по кастомному индексу.
    Это поможет модели понять, какие синонимы были более "близки" к таргету в корпусе.
    """
    syn_proximity = {}

    for syn in synonyms_filtered:
        syn_proximity[syn] = proximity_weights.get(syn, 0.0)
    
    syn_proximity_sorted = dict(sorted(syn_proximity.items(), key=lambda x: x[1], reverse=True))
    
    return syn_proximity_sorted

def proximity_neighbours_for_synonyms(synonyms_filtered, raw_data, decay_distance=0.95, decay_brks=0.85, decay_sents=0.9, stopwords=None):
    """
    Для каждого отфильтрованного синонима считаем его соседей по "Индексу ДИКС".
    Позволяет LLM понять, какие слова были близки к каждому синониму, а не только к таргету.
    """
    neighbours_for_synonyms = {}

    for syn in synonyms_filtered:
        weights = get_proximity_index_neighbors(raw_data, syn, decay_distance=decay_distance, decay_brks=decay_brks, decay_sents=decay_sents, stopwords=stopwords)
        # Сортируем и берем топ-10 соседей для каждого синонима
        neighbours_for_synonyms[syn] = weights.most_common(10)

    return neighbours_for_synonyms

prompt_prefix = '''
Ты — филолог-исследователь, работающий с корпусными данными поэтических текстов Заболоцкого. Твоя задача — написать краткий аналитический комментарий на основе предоставленных данных.

--- МЕТОДОЛОГИЯ И ТЕРМИНЫ ---
1. "Семантическая близость" (векторные синонимы в общем корпусе): близость слова к таргету в русской художественной литературе.
2. "Динамический индекс контекстуальной близости" (индекс ДИКС): частота и плотность нахождения слова рядом с таргетом в стихах. Учитывает переносы строки и строфы, границы строк (переход строки снижает вес связи) и границы предложений. Высокий индекс = слова "живут" в одном контексте.
3. Окружение синонимов: топ-10 соседей показывают тематическое поле, в котором этот синоним функционирует у автора.

--- ПРАВИЛА ОТВЕТА ---
- Пиши 4-5 коротких абзаца.
- Второй и последний абзац могут быть более длинными - предполгается, что они будут максимально содержательными и аналитичными.
- БЕЗ вступлений ("На основе данных...", "Ниже представлен анализ...").
- БЕЗ технических меток и сырых скобок с весами (пиши "индекс близости составляет 0.8", а не "(0.8)").
- ТОЛЬКО на основе предоставленных цифр. Не выдумывай факты биографии. Не добавляй те интерпретации, которые не следуют из данных.

--- ФОРМАТ ---
Абзац 1: Частотность и динамика (пики, годы, цифры).
Абзац 2: Синонимы в текстах (кто "дружит" с таргетом, а кто игнорируется), сравнение с ближайшими соседями по индексу ДИКС.
Абзац 3: Различия в окружении (сравнение лемм-соседей разных синонимов).
Абзац 4: Отсутствующие синонимы (что из общего языка Заболоцкий не взял).
Абзац 5: Главная аномалия или яркое наблюдение на основе чисел.
'''

def prepare_llm_prompt(target_word, synonyms, synonyms_filtered, syn_proximity, neighbors_for_synonyms, total_occurrences, year_dist, proximity_weights):
    """
    Формирует расширенный текстовый промпт для LLM.

    Включаем:
    - Список синонимов с их весами связи
    - Топ соседей для каждого синонима по "Индексу ДИКС"
    Это позволит модели делать более обоснованные выводы о том, какие синонимы были действительно релевантными в корпусе.
    """
    syn_blocks = []

    syn_proximity = dict(sorted(syn_proximity.items(), key=lambda x: x[1], reverse=True))

    # Формируем строку с динамикой по годам
    if year_dist:
        sorted_years = sorted(year_dist.items())
        year_dist_str = ", ".join([f"{year}: {count}" for year, count in sorted_years])
        peak_year = max(year_dist, key=year_dist.get)
        peak_count = year_dist[peak_year]
    else:
        year_dist_str = "нет данных"
        peak_year = None
        peak_count = 0

    for syn, neighbors in list(neighbors_for_synonyms.items())[:10]:
        neighbors_line = ", ".join([f"{n} ({w:.2f})" for n, w in neighbors])
        syn_blocks.append(f"  - '{syn}': {neighbors_line}")
    
    neighbors_for_synonyms_str = "\n".join(syn_blocks)
 
    synonyms_str = ", ".join([f"{syn} ({prox:.4f})" for syn, prox in syn_proximity.items()])

    # Сборка финального текста
    prompt = f"""
ОБЪЕКТ АНАЛИЗА: "{target_word.upper()}"

--- ДАННЫЕ ПО СЛОВУ ---
Всего употреблений: {total_occurrences}
Динамика: {year_dist_str}
{f'Пик: {peak_year} год ({peak_count} вхождений)' if peak_year else ''}

--- ВЕКТОРНЫЕ СИНОНИМЫ СЛОВА (ОБЩИЙ КОРПУС) ---
{', '.join([f"{syn} ({score:.4f})" for syn, score in synonyms])}

--- ВЕКТОРНЫЕ СИНОНИМЫ В КОРПУСЕ (С ИНДЕКСОМ КОНТЕКСТУАЛЬНОЙ БЛИЗОСТИ) ---
{synonyms_str if synonyms_str else 'нет'}

--- БЛИЖАЙШИЕ СОСЕДИ ПО ИНДЕКСУ ДИКС (ТОП-10) ---
{', '.join([f"{syn} ({prox:.4f})" for syn, prox in proximity_weights.most_common(10)])}

--- ОТСУТСТВУЮЩИЕ В ТЕКСТАХ СИНОНИМЫ ---
{', '.join([syn for syn, score in synonyms if syn not in synonyms_filtered]) or 'нет'}

--- ОКРУЖЕНИЕ СИНОНИМОВ (ТОП-10 СОСЕДЕЙ) ---
{neighbors_for_synonyms_str}
"""
    return prompt
