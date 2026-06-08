import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter

# --- Совместимость при запуске напрямую ---
# Позволяет запускать как `python src/map_builder.py` из корня проекта
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analyzer import get_proximity_index_neighbors
from src.utils import russian_stopwords

# ── Настройки ────────────────────────────────────────────────────────────────

AUTHOR_PRONOUNS = {"я", "мы", "вы", "наш", "ваш"}  # Авторские местоимения (не отсеиваются)
TOP_N_WORDS     = 305    # топ-300 авторских слов + 5 гарантированных местоимений
N_CLUSTERS      = 20     # количество кластеров K-Means
UMAP_N_NEIGHBORS = 25    # параметр UMAP: локальность структуры
UMAP_MIN_DIST   = 0.1   # параметр UMAP: минимальное расстояние между точками
RANDOM_STATE    = 42

# Параметры затухания Индекса ДИКС (те же, что по умолчанию в app.py)
DECAY_DISTANCE  = 0.95
DECAY_BRKS      = 0.85
DECAY_SENTS     = 0.9

DATABASE_PATH   = os.path.join("data", "database.parquet")
OUTPUT_PATH     = os.path.join("data", "word_cluster_map.json")

# ── Вспомогательные функции ───────────────────────────────────────────────────

def load_corpus(database_path: str) -> list[dict]:
    """
    Читает database.parquet и собирает список текстов в формате,
    ожидаемом функцией get_proximity_index_neighbors.
    """
    print("Загружаю корпус...")
    df = pd.read_parquet(database_path)
    corpus = []

    for _, row in df.iterrows():
        try:
            # В parquet данные уже хранятся как списки/массивы
            formatted_sentences      = list(row["formatted_sentences"]) if row["formatted_sentences"] is not None else []
            lemmas_separated         = list(row["lemmas_separated"]) if row["lemmas_separated"] is not None else []
            lemmas_cleaned           = list(row["lemmas_cleaned"]) if row["lemmas_cleaned"] is not None else []
            lemmas_pos_tagged        = list(row["lemmas_pos_tagged"]) if row["lemmas_pos_tagged"] is not None else []

            corpus.append({
                "title":               str(row.get("title", "")),
                "year_finished":       int(row["year_finished"]) if pd.notnull(row.get("year_finished")) else 0,
                "raw_text":            str(row.get("raw_text", "")),
                "formatted_sentences": formatted_sentences,
                "lemmas_separated":    lemmas_separated,
                "lemmas_cleaned":      lemmas_cleaned,
                "lemmas_pos_tagged":   lemmas_pos_tagged,
            })
        except Exception as e:
            print(f"  ⚠️  Пропускаю строку (ошибка парсинга): {e}")
            continue

    print(f"  ✅ Загружено текстов: {len(corpus)}")
    return corpus


def get_top_words(corpus: list[dict], top_n: int, stopwords: set) -> list[str]:
    """
    Считает частоту лемм по всему корпусу и возвращает топ-N
    без стоп-слов, разделителей и однобуквенных токенов.

    ИСКЛЮЧЕНИЕ: авторские местоимения (я, мы, вы, наш, ваш) НЕ отсеиваются!
    """

    print(f"Определяю топ-{top_n} слов корпуса (+ авторские местоимения)...")
    freq: Counter = Counter()

    for item in corpus:
        for sent in item["lemmas_cleaned"]:
            for lemma in sent:
                if (
                    lemma
                    and lemma != "_BRK_"
                    and lemma.isalpha()
                    # ← КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: авторские местоимения НЕ отсеиваются!
                    and (lemma not in stopwords or lemma in AUTHOR_PRONOUNS)
                ):
                    freq[lemma] += 1

    # Берём топ-N кандидатов (+20 для подстраховки)
    top_words = [word for word, _ in freq.most_common(top_n + 20)]

    # Гарантируем включение авторских местоимений
    for pronoun in AUTHOR_PRONOUNS:
        if pronoun not in top_words:
            # Добавляем местоимение в начало
            top_words.insert(0, pronoun)

    # Обрезаем до нужного размера
    top_words = top_words[:top_n]

    print(f"  ✅ Топ-{top_n} определён (топ-300 авторских + 5 местоимений). Первые 10: {top_words[:10]}")
    return top_words


def build_proximity_matrix(
    corpus: list[dict],
    top_words: list[str],
    stopwords: set,
    decay_distance: float,
    decay_brks: float,
    decay_sents: float,
) -> np.ndarray:
    """
    Для каждого слова из top_words вызывает get_proximity_index_neighbors
    и заполняет строку матрицы весов N×N.

    Матрица симметризуется усреднением: M_sym = (M + M.T) / 2
    """
    n = len(top_words)
    word_to_idx = {w: i for i, w in enumerate(top_words)}
    matrix = np.zeros((n, n), dtype=np.float32)

    print(f"\nСтроим матрицу {n}×{n} — это займёт время...")
    print("  Прогресс: ", end="", flush=True)

    for i, word in enumerate(top_words):
        weights = get_proximity_index_neighbors(
            filtered_corpus=corpus,
            target_norm=word,
            decay_distance=decay_distance,
            decay_brks=decay_brks,
            decay_sents=decay_sents,
            stopwords=stopwords,
        )

        for neighbor_word, weight in weights.items():
            j = word_to_idx.get(neighbor_word)
            if j is not None:
                matrix[i, j] = float(weight)

        # Прогресс-бар каждые 10 слов
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"{i + 1}/{n}", end=" ", flush=True)

    print("\n  Симметризую матрицу...")
    matrix = (matrix + matrix.T) / 2.0
    print("  ✅ Матрица построена.")
    return matrix


def reduce_umap(matrix: np.ndarray, n_neighbors: int, min_dist: float, random_state: int) -> np.ndarray:
    """Понижает размерность матрицы до 2D с помощью UMAP."""
    try:
        import umap
    except ImportError:
        raise ImportError(
            "Библиотека umap-learn не найдена.\n"
            "Установите: pip install umap-learn"
        )

    print("Запускаю UMAP...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",        # косинусное расстояние подходит для весовых векторов
        random_state=random_state,
        n_components=2,
    )
    embedding = reducer.fit_transform(matrix)
    print("  ✅ UMAP завершён.")
    return embedding


def cluster_kmeans(embedding: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    """Кластеризует 2D-эмбеддинг методом K-Means."""
    from sklearn.cluster import KMeans

    print(f"Кластеризую методом K-Means (k={n_clusters})...")
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(embedding)
    print("  ✅ Кластеризация завершена.")
    return labels


def save_map(
    output_path: str,
    top_words: list[str],
    embedding: np.ndarray,
    labels: np.ndarray,
    freq_counter: Counter,
) -> None:
    """
    Сохраняет результат в JSON для последующей загрузки в Streamlit.
    Добавляет поле top_position — позицию слова в топе по частотности (1-based).
    При одинаковой частоте сортируем по алфавиту.
    """
    # Создаём промежуточные данные
    word_indices = {word: i for i, word in enumerate(top_words)}
    word_data = []

    for word in top_words:
        word_data.append({
            "word": word,
            "freq": freq_counter.get(word, 0),
            "idx": word_indices[word],  # индекс в исходном списке (для координат)
        })

    # Сортируем по частоте (убывание) и по слову (возрастание)
    word_data_sorted = sorted(word_data, key=lambda x: (-x["freq"], x["word"]))

    # Присваиваем позиции (1-based)
    records = []
    for top_pos, item in enumerate(word_data_sorted, start=1):
        idx = item["idx"]
        word = item["word"]

        records.append({
            "word":         word,
            "x":            float(embedding[idx, 0]),
            "y":            float(embedding[idx, 1]),
            "cluster":      int(labels[idx]) + 1,  # +1: кластеры начинаются с 1
            "freq":         int(item["freq"]),
            "top_position": top_pos,  # позиция в топе (1-based)
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Карта сохранена: {output_path} ({len(records)} слов, кластеры 1-{int(labels.max()) + 1})")


# ── Главная функция ────────────────────────────────────────────────────────────

def build_word_cluster_map(
    database_path: str  = DATABASE_PATH,
    output_path: str    = OUTPUT_PATH,
    top_n: int          = TOP_N_WORDS,
    n_clusters: int     = N_CLUSTERS,
    decay_distance: float = DECAY_DISTANCE,
    decay_brks: float   = DECAY_BRKS,
    decay_sents: float  = DECAY_SENTS,
    umap_n_neighbors: int = UMAP_N_NEIGHBORS,
    umap_min_dist: float  = UMAP_MIN_DIST,
    random_state: int   = RANDOM_STATE,
) -> None:
    """
    Точка входа. Вызывается из командной строки или из другого модуля.
    Все шаги: загрузка → топ слов → матрица → UMAP → K-Means → сохранение.
    """

    # 1. Загрузка корпуса
    corpus = load_corpus(database_path)
    if not corpus:
        print("❌ Корпус пустой. Проверьте путь к database.parquet.")
        return

    # 2. Топ-N слов
    top_words = get_top_words(corpus, top_n, russian_stopwords)
    if len(top_words) < 2:
        print("❌ Слишком мало слов для построения карты.")
        return

    # ✅ ПРОВЕРКА: должно быть ровно top_n слов
    assert len(top_words) == top_n, f"❌ ОШИБКА: получилось {len(top_words)} слов вместо {top_n}!"
    print(f"  ✅ Проверка пройдена: ровно {len(top_words)} слов в топе")

    # Частоты нужны для размера точек на карте
    # ВАЖНО: используем ТОТЖЕ ФИЛЬТР, что и в get_top_words()
    freq: Counter = Counter()
    for item in corpus:
        for sent in item["lemmas_cleaned"]:
            for lemma in sent:
                if (
                    lemma
                    and lemma != "_BRK_"
                    and lemma.isalpha()
                    # Тот же фильтр: авторские местоимения НЕ отсеиваются!
                    and (lemma not in russian_stopwords or lemma in AUTHOR_PRONOUNS)
                ):
                    freq[lemma] += 1

    # 3. Матрица Индексу ДИКС
    matrix = build_proximity_matrix(
        corpus, top_words, russian_stopwords,
        decay_distance, decay_brks, decay_sents,
    )

    # 4. UMAP → 2D
    embedding = reduce_umap(matrix, umap_n_neighbors, umap_min_dist, random_state)

    # 5. Кластеризация
    labels = cluster_kmeans(embedding, n_clusters, random_state)

    # 6. Сохранение
    save_map(output_path, top_words, embedding, labels, freq)


if __name__ == "__main__":
    build_word_cluster_map()