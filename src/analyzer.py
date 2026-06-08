import os
import re
import json
from collections import Counter
from navec import Navec
from src.utils import read_text_file, get_sentences, ms, count_words

path = os.path.join('models', 'navec_hudlit_v1_12B_500K_300d_100q.tar')
navec = Navec.load(path)

POS_MYSTEM = {
    'A': 'Прилагательное', 'ADV': 'Наречие', 'ADVPRO': 'Местоименное наречие',
    'ANUM': 'Числительное-прилагательное', 'APRO': 'Местоимение-прилагательное',
    'COM': 'Часть композита - сложного слова', 'CONJ': 'Союз', 'INTJ': 'Междометие',
    'NUM': 'Числительное', 'PART': 'Частица', 'PR': 'Предлог', 'S': 'Существительное',
    'SPRO': 'Местоимение-существительное', 'V': 'Глагол',
    #КАСТОМНЫЕ ТЕГИ
    'HYPHCOMP': 'Сложносоставные слова', 'UNK': 'Неизвестно'
}

# --- ФУНКЦИИ ДЛЯ АНАЛИЗА СЛОВА ---

def highlight_lemma_forms_in_text(raw_text, vocabulary_forms, case_insensitive=True):
    """
    Подсвечивает все встреченные словоформы целевой леммы в тексте.
    Использует специальный маркер вместо звёздочек для совместимости со Streamlit.

    Args:
        raw_text: Исходный текст
        vocabulary_forms: Список всех словоформ целевой леммы
        case_insensitive: Игнорировать ли регистр при поиске

    Returns:
        Текст с маркированными формами (для последующей обработки в UI)
    """
    if not vocabulary_forms:
        return raw_text

    # Сортируем по длине (длинные первыми) чтобы избежать partial matches
    sorted_forms = sorted(set(vocabulary_forms), key=len, reverse=True)
    result = raw_text

    for form in sorted_forms:
        if case_insensitive:
            # Case-insensitive search with word boundaries
            pattern = r'\b' + re.escape(form) + r'\b'
            # Используем специальный маркер для выделения
            result = re.sub(pattern, f'<<<{form}>>>', result, flags=re.IGNORECASE)
        else:
            pattern = r'\b' + re.escape(form) + r'\b'
            result = re.sub(pattern, f'<<<{form}>>>', result)

    return result


# 1. Функция контекста и общей статистики
def find_all_form_occurrences(text, forms):
    """
    Находит все вхождения словоформ в тексте (case-insensitive).
    Возвращает список кортежей (form, start_pos, end_pos, char_position)
    в порядке появления в тексте.
    """
    occurrences = []
    # Сортируем по длине (длинные первыми) чтобы избежать partial matches
    sorted_forms = sorted(set(forms), key=len, reverse=True)

    for form in sorted_forms:
        pattern = r'\b' + re.escape(form) + r'\b'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            occurrences.append((form, match.start(), match.end()))

    # Сортируем по позиции в тексте
    return sorted(occurrences, key=lambda x: x[1])


def get_occurrence_data(filtered_corpus, target_norm, lemma_forms):
    """
    Ищет все вхождения словоформ целевой леммы в formatted_sentences.
    Для каждого вхождения:
    - Выводит предложение целиком как контекст
    - Если < 3 слов, добавляет самое короткое соседнее предложение
    - Подсвечивает все найденные формы
    - Считает вхождения по годам

    Возвращает:
    - total_occurrences: total count
    - contexts: список словарей для таблицы
    - year_dist: Counter по годам
    """
    contexts = []
    year_dist = Counter()
    total_occurrences = 0

    target_forms = lemma_forms.get(target_norm, [])

    if not target_forms:
        return 0, [], year_dist

    for item in filtered_corpus:
        sentences = item['formatted_sentences']

        # Для каждого предложения ищем вхождения
        for sent_idx, sentence in enumerate(sentences):
            occurrences = find_all_form_occurrences(sentence, target_forms)

            if not occurrences:
                continue

            # Есть вхождения — добавляем контекст
            # Если предложение < 3 слов, пытаемся добавить самое короткое соседнее
            context_sentence = sentence.strip(' / —–-')

            if count_words(context_sentence, remove_punct=False) < 3:
                neighbor_sentences = []

                if sent_idx > 0:
                    prev_sentence = sentences[sent_idx - 1].strip().strip(' / —–-')
                    neighbor_sentences.append((prev_sentence, count_words(prev_sentence, remove_punct=False)))

                if sent_idx < len(sentences) - 1:
                    next_sentence = sentences[sent_idx + 1].strip().strip(' / —–-')
                    neighbor_sentences.append((next_sentence, count_words(next_sentence, remove_punct=False)))

                if neighbor_sentences:
                    # Берём самое короткое из соседних
                    shortest_neighbor = min(neighbor_sentences, key=lambda x: x[1])[0]
                    context_sentence = f"{context_sentence} {shortest_neighbor}".strip(' / —–-')

            # Подсвечиваем все словоформы в контексте
            context_sentence = context_sentence[:1].upper() + context_sentence[1:] if context_sentence else context_sentence
            display_context = highlight_lemma_forms_in_text(context_sentence, target_forms)

            # Для каждого вхождения в этом предложении добавляем отдельную строку
            for form, _, _ in occurrences:
                total_occurrences += 1
                year_dist[item['year_finished']] += 1

                contexts.append({
                    "Контекст": display_context,
                    "Произведение": item['title'],
                    "Год": item['year_finished']
                })

    return total_occurrences, contexts, year_dist

# 2. Функция классического окна (без индексов)
def get_window_neighbors(raw_data, target_norm, window_size, stopwords=None):
    """
    Считает соседей в жестком окне и их части речи, используя lemmas_pos_tagged.
    Находит позиции целевого слова в lemmas_pos_tagged.
    Парсит lemmas_pos_tagged в формате "lemma/POS" и используюет POS-теги от MyStem.

    Возвращает два варианта:
    - Без стопслов: neighbors, neighbor_pos
    - Со стопсловами: neighbors_w_stwords, neighbors_pos_stwords
    """
    neighbors = Counter()
    neighbor_pos = Counter()
    neighbors_w_stwords = Counter()
    neighbors_pos_stwords = Counter()

    for text in raw_data:
        # Формируем плоское представление из lemmas_pos_tagged
        flat_lemmas_pos = []
        for sent_lemmas_pos in text['lemmas_pos_tagged']:
            flat_lemmas_pos.extend(sent_lemmas_pos)

        # Парсим каждый элемент "lemma/POS" и отделяем леммы и POS-теги
        flat_lemmas = []
        flat_pos_tags = []
        for lemma_pos_pair in flat_lemmas_pos:
            if '/' in lemma_pos_pair:
                lemma, pos_tag = lemma_pos_pair.rsplit('/', 1)
                flat_lemmas.append(lemma)
                flat_pos_tags.append(pos_tag)
            else:
                flat_lemmas.append(lemma_pos_pair)
                flat_pos_tags.append('UNK')

        # Находим все позиции целевого слова
        target_positions = [idx for idx, lemma in enumerate(flat_lemmas) if lemma == target_norm]

        # Для каждого вхождения берём соседей в окне
        for targ_idx in target_positions:
            start = max(0, targ_idx - window_size)
            end = min(len(flat_lemmas), targ_idx + window_size + 1)

            for j in range(start, end):
                if j == targ_idx:
                    continue
                lemma = flat_lemmas[j]
                if lemma and lemma[0].isalpha():
                    pos_tag = flat_pos_tags[j]
                    pos_name = POS_MYSTEM.get(pos_tag, 'Другое')

                    # Добавляем в версию со стопсловами (без фильтрации)
                    neighbors_w_stwords[lemma] += 1
                    neighbors_pos_stwords[pos_name] += 1

                    # Добавляем в версию без стопслов (с фильтрацией)
                    if lemma not in (stopwords or []):
                        neighbors[lemma] += 1
                        neighbor_pos[pos_name] += 1

    return neighbors, neighbor_pos, neighbors_w_stwords, neighbors_pos_stwords

# 3. Функция "Индекса ДИКС" (динамический индекс контекстуальной близости)
def get_proximity_index_neighbors(filtered_corpus, target_norm, decay_distance, decay_brks, decay_sents, stopwords=None):
    """
    Для каждого вхождения таргета сканируем весь текст и считаем вес связи с каждой леммой, учитывая:
    - Дистанцию в словах (чем дальше, тем слабее связь)
    - Количество разрывов строк (_BRK_) на пути (каждый разрыв ослабляет связь)
    - Количество границ предложений (точек) на пути (каждая граница ослабляет связь)

    СИНХРОНИЗАЦИЯ ИНДЕКСОВ:
    - lemmas_pos_tagged: чистые леммы "lemma/POS" без маркеров (для анализа)
    - lemmas_separated: леммы со счетом границ _BRK_ (для определения разрывов строк)

    Оба массива парсятся параллельно для синхронизации позиций.
    """
    weights = Counter()

    for item in filtered_corpus:
        # 1. Парсим оба источника параллельно для синхронизации индексов
        flat_lemmas = []
        flat_lemmas_separated = []  # Отслеживаем _BRK_ маркеры
        mapping_clean_to_separated = []  # Индекс: clean_idx -> separated_idx
        sent_boundaries_clean = []

        curr_idx_clean = 0
        curr_idx_separated = 0

        for sent_pos_tagged, sent_separated in zip(item['lemmas_pos_tagged'], item['lemmas_separated']):
            sent_lemmas = []
            sep_idx = 0  # Позиция в текущем предложении lemmas_separated

            for lemma_pos_pair in sent_pos_tagged:
                # Парсим "lemma/POS" для получения леммы
                if '/' in lemma_pos_pair:
                    lemma = lemma_pos_pair.rsplit('/', 1)[0]
                else:
                    lemma = lemma_pos_pair

                sent_lemmas.append(lemma)
                flat_lemmas.append(lemma)

                # Синхронизируем с lemmas_separated (пропускаем _BRK_)
                while sep_idx < len(sent_separated) and sent_separated[sep_idx] == '_BRK_':
                    flat_lemmas_separated.append('_BRK_')
                    sep_idx += 1

                if sep_idx < len(sent_separated):
                    flat_lemmas_separated.append(sent_separated[sep_idx])
                    mapping_clean_to_separated.append(curr_idx_separated)
                    curr_idx_separated += 1
                    sep_idx += 1

                curr_idx_clean += 1

            # Добавляем оставшиеся _BRK_ в конце предложения
            while sep_idx < len(sent_separated):
                if sent_separated[sep_idx] == '_BRK_':
                    flat_lemmas_separated.append('_BRK_')
                curr_idx_separated += 1
                sep_idx += 1

            sent_boundaries_clean.append(curr_idx_clean)

        # 2. Находим все позиции целевого слова в чистых леммах
        target_positions = [idx for idx, lemma in enumerate(flat_lemmas) if lemma == target_norm]

        # 3. Для каждого вхождения таргета сканируем весь текст
        for t_idx in target_positions:
            for s_idx, lemma in enumerate(flat_lemmas):

                if s_idx == t_idx or not lemma or not lemma[0].isalpha():
                    continue

                if stopwords and lemma in stopwords:
                    continue

                # Расстояние в словах (без маркеров)
                d = abs(s_idx - t_idx)

                # Находим соответствующие индексы в flat_lemmas_separated
                if s_idx < len(mapping_clean_to_separated) and t_idx < len(mapping_clean_to_separated):
                    sep_start = mapping_clean_to_separated[min(t_idx, s_idx)]
                    sep_end = mapping_clean_to_separated[max(t_idx, s_idx)]

                    # Сколько разрывов строк (_BRK_) встретилось на пути
                    fragment_separated = flat_lemmas_separated[sep_start:sep_end + 1]
                    n_brks = fragment_separated.count('_BRK_')
                else:
                    n_brks = 0

                # Сколько границ предложений пересекли
                n_sents = len([b for b in sent_boundaries_clean if min(t_idx, s_idx) < b <= max(t_idx, s_idx)])

                # --- ИТОГОВЫЙ ВЕС СВЯЗИ ---
                weight = (decay_distance ** d) * (decay_brks ** n_brks) * (decay_sents ** n_sents)
                weights[lemma] += weight

    return weights

# 4. Анализ дельты между двумя периодами
def calculate_delta_analysis(results_1, results_2, count_stopwords=False):
    """
    Сравнивает два периода по ДИНАМИЧЕСКОМУ ИНДЕКСУ контекстуальной близости.
    Анализирует только proximity_weights — статистику контекстного окна игнорирует.

    Возвращает словарь:
    {
        'occurrences_delta': int,
        'occurrences_pct': float,
        'appeared_words': [(word, index_2), ...],  # Слова, появившиеся во втором периоде
        'disappeared_words': [(word, index_1), ...],  # Слова, исчезнувшие
        'changed_words': [
            {
                'word': str,
                'index_1': float, 'index_2': float, 'index_delta': float, 'index_pct': float,
                'status': str  # 'growing', 'declining', 'stable'
            },
            ...
        ],
        'top_rising': [...],     # Топ-5 растущих слов по индексу
        'top_declining': [...]   # Топ-5 падающих слов по индексу
    }
    """
    if not results_1 or not results_2:
        return None

    weights_1 = results_1['proximity_weights']
    weights_2 = results_2['proximity_weights']

    occurrences_1 = results_1['total_occurrences']
    occurrences_2 = results_2['total_occurrences']
    occurrences_delta = occurrences_2 - occurrences_1
    occurrences_pct = (occurrences_delta / max(occurrences_1, 1)) * 100

    # Словари слов в каждом периоде
    words_1 = set(weights_1.keys())
    words_2 = set(weights_2.keys())

    # Категоризация
    appeared = words_2 - words_1
    disappeared = words_1 - words_2
    both_periods = words_1 & words_2

    appeared_words = [
        (word, weights_2.get(word, 0.0))
        for word in sorted(appeared, key=lambda w: weights_2.get(w, 0.0), reverse=True)
    ]

    disappeared_words = [
        (word, weights_1.get(word, 0.0))
        for word in sorted(disappeared, key=lambda w: weights_1.get(w, 0.0), reverse=True)
    ]

    # Слова, которые были в обоих периодах — анализируем дельту
    changed_words = []
    for word in both_periods:
        index_1 = weights_1.get(word, 0.0)
        index_2 = weights_2.get(word, 0.0)

        index_delta = index_2 - index_1
        index_pct = (index_delta / max(index_1, 0.001)) * 100 if index_1 > 0 else (100 if index_2 > 0 else 0)

        # Определяем статус
        if index_delta > 0:
            status = 'growing'
        elif index_delta < 0:
            status = 'declining'
        else:
            status = 'stable'

        changed_words.append({
            'word': word,
            'index_1': index_1,
            'index_2': index_2,
            'index_delta': index_delta,
            'index_pct': index_pct,
            'status': status,
        })

    # Сортируем и подбираем топ
    changed_by_index_growth = sorted(
        [w for w in changed_words if w['index_delta'] > 0],
        key=lambda x: x['index_delta'],
        reverse=True
    )[:5]

    changed_by_index_decline = sorted(
        [w for w in changed_words if w['index_delta'] < 0],
        key=lambda x: x['index_delta']
    )[:5]

    return {
        'occurrences_delta': occurrences_delta,
        'occurrences_pct': occurrences_pct,
        'appeared_words': appeared_words[:10],
        'disappeared_words': disappeared_words[:10],
        'changed_words': sorted(changed_words, key=lambda x: abs(x['index_delta']), reverse=True)[:15],
        'top_rising': changed_by_index_growth,
        'top_declining': changed_by_index_decline
    }

# 5. Главная координирующая функция
def full_word_analysis(filtered_corpus, target_word, window_size=5, decay_distance=0.95, decay_brks=0.85, decay_sents=0.9, stopwords=None, lemma_forms=None):

    if lemma_forms is None:
        lemma_forms = {}

    # Шаг 1: Находим все вхождения слова в formatted_sentences и собираем контексты
    total_occurrences, contexts, year_dist = get_occurrence_data(filtered_corpus, target_word, lemma_forms)

    if not filtered_corpus or not contexts:
        return None  # Если в корпусе нет данных, возвращаем None

    # Шаг 2: Классическое окно контекстов
    neighbors, neighbor_pos, neighbors_w_stwords, neighbors_pos_stwords = get_window_neighbors(filtered_corpus, target_word, window_size, stopwords)

    # Шаг 3: Динамический индекс для всех слов в тексте
    proximity_weights = get_proximity_index_neighbors(filtered_corpus, target_word, decay_distance, decay_brks, decay_sents, stopwords)

    return {
        'total_occurrences': total_occurrences,
        'contexts': contexts,
        'year_dist': year_dist,
        'window_neighbors': {
            'filtered': neighbors,
            'with_stopwords': neighbors_w_stwords
        },
        'pos_dist': {
            'filtered': neighbor_pos,
            'with_stopwords': neighbors_pos_stwords
        },
        'proximity_weights': proximity_weights
    }

# --- ФУНКЦИИ ДЛЯ СИНОНИМОВ И ФИЛЬТРАЦИИ ---

def get_unique_synonyms(target_word, top_n_to_return=20, search_depth=50):
    """
    1. Находит глубокий топ синонимов (50 шт).
    2. Приводит их к нормальной форме (лемме).
    3. Оставляет только уникальные леммы, отличные от самого target_word.
    4. Возвращает срез нужной длины.
    """
    if target_word not in navec:
        return []

    raw_sims = []
    for word in navec.vocab.words:
        if not word.isalpha():
            continue
        score = navec.sim(target_word, word)
        raw_sims.append((word, score))
    
    raw_sims.sort(key=lambda x: x[1], reverse=True)
    
    # 2. Фильтрация через лемматизацию
    unique_lemmas = []
    seen_lemmas = {target_word.lower()} # Сразу игнорируем само искомое слово
    
    for word, score in raw_sims:
        # Лемматизируем кандидата

        lemma = ms.analyze(word)[0]['analysis'][0]['lex']
        
        if lemma not in seen_lemmas:
            unique_lemmas.append((lemma, score))
            seen_lemmas.add(lemma)
            
        # Как только набрали нужное количество уникальных понятий — выходим
        if len(unique_lemmas) >= search_depth:
            break

    # 3. Возвращаем срез (20 или сколько запрошено)
    final_cutoff = min(len(unique_lemmas), top_n_to_return)
    return unique_lemmas[:final_cutoff]

def filter_synonyms_by_corpus(synonyms):
    """
    Дополнительная фильтрация синонимов через корпус.
    Оставляет только те, которые реально встречаются в текстах.
    Проверяет и по ключам (леммам), и по самим записям (словоформам) в vocabulary_forms.json.
    """
    forms_path = os.path.join('data', 'vocabulary_forms.json')
    
    if not os.path.exists(forms_path):
        return []
    
    try:
        with open(forms_path, 'r', encoding='utf-8') as f:
            vocab_forms = json.load(f)
    except Exception:
        return []
    
    vocab_set = set(vocab_forms.keys())
    for word_forms in vocab_forms.values():
        if isinstance(word_forms, (list, set)):
            vocab_set.update(word_forms)
    
    filtered_synonyms = [syn for syn, score in synonyms if syn in vocab_set]
    
    return filtered_synonyms
