import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import re
import numpy as np
import altair as alt
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from collections import Counter, defaultdict
from src.utils import russian_stopwords
from src.prompt import prepare_llm_prompt, prompt_prefix
from src.analyzer import navec, full_word_analysis, get_unique_synonyms, filter_synonyms_by_corpus, calculate_delta_analysis, calculate_diachronic_log_likelihood
from src.prompt import proximity_neighbours_for_synonyms, synonyms_proximity_index
from dotenv import load_dotenv

load_dotenv()
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
cluster_maps_interpr_link = os.getenv("CLUSTER_MAPS_INTERPR_LINK")

@st.cache_data
def load_data():
    df = pd.read_parquet('data/database.parquet')
    return df.to_dict('records')  # Список словарей

@st.cache_data
def load_lemma_forms():
    """
    Загружает словарь 'лемма -> все встреченные словоформы'.
    Кешируется через st.cache_data для использования в приложении.
    """
    forms_path = os.path.join('data', 'vocabulary_forms.json')

    if os.path.exists(forms_path):
        try:
            with open(forms_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Ошибка при загрузке словаря словоформ: {e}")
            return {}
    else:
        st.error(f"Файл {forms_path} не найден. Запустите препроцессинг.")
        return {}

@st.cache_data
def load_cluster_map():
    """
    Загружает предрассчитанную семантическую карту (индекс ДИКС + UMAP + K-Means).
    Возвращает pd.DataFrame или None, если файл не существует.
    """
    map_path = os.path.join('data', 'word_cluster_map.json')
    if not os.path.exists(map_path):
        return None
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Ошибка при загрузке семантической карты: {e}")
        return None
    
@st.cache_data
def load_cluster_map_first_period():
    """
    Загружает предрассчитанную семантическую карту (индекс ДИКС + UMAP + K-Means).
    Возвращает pd.DataFrame или None, если файл не существует.
    """
    map_path = os.path.join('data', 'word_map_1918_1938.json')
    if not os.path.exists(map_path):
        return None
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Ошибка при загрузке семантической карты: {e}")
        return None

@st.cache_data
def load_cluster_map_last_period():
    """
    Загружает предрассчитанную семантическую карту (индекс ДИКС + UMAP + K-Means).
    Возвращает pd.DataFrame или None, если файл не существует.
    """
    map_path = os.path.join('data', 'word_map_1946_1958.json')
    if not os.path.exists(map_path):
        return None
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Ошибка при загрузке семантической карты: {e}")
        return None

@st.cache_data
def load_mayak_hapax():
    """
    Загружает hapax legomena Заболоцкого (слова только в его творчестве).
    Возвращает dict с metadata и списком гапаксов, или None если файл не существует.
    """
    hapax_path = os.path.join('data', 'mayak_hapax.json')
    if not os.path.exists(hapax_path):
        return None
    try:
        with open(hapax_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Ошибка при загрузке гапаксов: {e}")
        return None

@st.cache_data
def get_hapax_legomena(corpus_records, hapax_set):
    """
    Фильтрует гапаксы, которые встречаются в корпусе Заболоцкого ровно один раз.
    Исключает слова содержащие цифры.
    Возвращает dict {лемма: частота} для гапаксов с freq=1.
    """
    freq_counter = Counter()
    for item in corpus_records:
        for sentence in item['lemmas_cleaned']:
            freq_counter.update(sentence)

    # Фильтруем только гапаксы которые встречаются один раз и не содержат цифры
    hapax_once = {
        lemma: 1
        for lemma in hapax_set
        if freq_counter.get(lemma, 0) == 1 and not any(c.isdigit() for c in lemma)
    }
    return hapax_once

@st.cache_data
def check_hapax_in_navec(hapax_list):
    """
    Проверяет какие слова из списка не присутствуют в модели navec.
    Использует navec.vocab для проверки: если получаем unk_id, слова нет.
    Возвращает два списка: (слова_в_navec, слова_не_в_navec)
    """
    in_navec = []
    not_in_navec = []
    unk_id = navec.vocab.unk_id

    for lemma in hapax_list:
        # Используем navec.vocab.get() чтобы получить ID слова
        # Если слово неизвестно, вернётся unk_id
        word_id = navec.vocab.get(lemma, unk_id)

        if word_id != unk_id:
            in_navec.append(lemma)
        else:
            not_in_navec.append(lemma)

    return in_navec, not_in_navec


# --- Функции для статистики корпуса ---

@st.cache_data
def compute_general_metrics(corpus_records):
    """Вычисляет базовые метрики корпуса: тексты, токены, леммы, уникальные леммы."""
    total_texts = len(corpus_records)
    total_tokens = sum(len(item['tokens']) for item in corpus_records)
    all_lemmas = []
    for text in corpus_records:
        for sentence in text['lemmas_cleaned']:
            all_lemmas.extend(sentence)
    total_lemmas = len(all_lemmas)
    unique_lemmas = len(set(all_lemmas))
    texts_by_year = Counter(item['year_finished'] for item in corpus_records)
    return {
        'total_texts': total_texts,
        'total_tokens': total_tokens,
        'total_lemmas': total_lemmas,
        'unique_lemmas': unique_lemmas,
        'texts_by_year': dict(texts_by_year),
    }

@st.cache_data
def compute_frequency_dict(corpus_records, exclude_stopwords=True):
    """Возвращает Counter лемм по корпусу."""
    counter = Counter()
    for item in corpus_records:
        for sentence in item['lemmas_cleaned']:
            counter.update(sentence)
    if exclude_stopwords:
        for sw in russian_stopwords:
            if sw in counter:
                del counter[sw]
    return counter

@st.cache_data
def compute_vocabulary_growth(corpus_records):
    """Считает рост словарного запаса (накопленных уникальных лемм) по годам."""
    by_year = defaultdict(list)
    for item in corpus_records:
        for sentence in item['lemmas_cleaned']:
            by_year[item['year_finished']].extend(sentence)

    sorted_years = sorted(by_year.keys())
    seen_lemmas = set()
    growth_data = []
    for year in sorted_years:
        lemmas_year = by_year[year]
        new_lemmas_count = len(set(lemmas_year) - seen_lemmas)
        seen_lemmas.update(lemmas_year)
        unique_in_year = len(set(lemmas_year))
        total_in_year = len(lemmas_year)
        growth_data.append({
            'Год': year,
            'Уникальных лемм накоплено': len(seen_lemmas),
            'Новых лемм': new_lemmas_count,
            'Всего лемм в году': total_in_year,
            'Type-Token Ratio': round(unique_in_year / total_in_year, 3) if total_in_year > 0 else 0,
        })
    return growth_data

@st.cache_data
def cached_get_unique_synonyms(word, top_n=20, depth=50):
    return get_unique_synonyms(word, top_n_to_return=top_n, search_depth=depth)

@st.cache_data
def compute_vector_map(corpus_records, top_n=100, exclude_stopwords=True, pca_base_size=300):
    """
    Возвращает DataFrame с 2D-координатами (PCA) для топ-N лемм корпуса.

    ⚠️  PCA вычисляется на pca_base_size словах, затем берётся подмножество top_n.
    Это гарантирует статичные координаты независимо от top_n!
    """
    # Гарантируем, что pca_base_size >= top_n
    if pca_base_size < top_n:
        pca_base_size = top_n

    freq_counter = compute_frequency_dict(corpus_records, exclude_stopwords=exclude_stopwords)
    # Берём pca_base_size слов для вычисления PCA — гарантирует стабильные координаты
    all_words_for_pca = [word for word, _ in freq_counter.most_common(pca_base_size)]

    words_in_navec = [(word, freq_counter[word]) for word in all_words_for_pca if word in navec]
    if len(words_in_navec) < 3:
        return None

    words, freqs = zip(*words_in_navec)
    matrix = np.array([navec[w] for w in words])

    # PCA через numpy (вычисляется на всех словах из pca_base_size)
    centered = matrix - matrix.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ Vt[:2].T

    # Берём первые top_n из pca_base_size вычисленных координат
    return pd.DataFrame({
        'Слово': list(words)[:top_n],
        'x': coords[:top_n, 0],
        'y': coords[:top_n, 1],
        'Частота': list(freqs)[:top_n],
    })

def build_mayak_semantic_map(df: pd.DataFrame, selected_clusters: list[int], size_by_freq: bool):
    """
    Строит интерактивный scatter-plot Plotly для семантической карты (индекс ДИКС).
    Каждый кластер — отдельный trace для независимого включения/отключения в легенде.
    """
    fig = go.Figure()

    # Палитра для кластеров (20 кластеров)
    cluster_palette = [
        "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
        "#9B5DE5", "#00BBF9", "#F15BB5", "#06D6A0", "#FB5607",
        "#8338EC", "#3A86FF", "#FFBE0B", "#FF006E", "#8AC926",
        "#DC2F02", "#370617", "#03071E", "#D62828", "#F77F00",
    ]

    all_clusters = sorted(df["cluster"].unique())

    for cluster_id in all_clusters:
        if cluster_id not in selected_clusters:
            continue

        subset = df[df["cluster"] == cluster_id]
        color = cluster_palette[(cluster_id - 1) % len(cluster_palette)]

        if size_by_freq:
            # Нормируем размер точки: минимум 8, максимум 28
            freq_vals = subset["freq"].values.astype(float)
            max_f = freq_vals.max() if freq_vals.max() > 0 else 1
            sizes = 8 + (freq_vals / max_f) * 20
        else:
            sizes = 14

        fig.add_trace(go.Scatter(
            x=subset["x"],
            y=subset["y"],
            mode="markers+text",
            name=f"Кластер {cluster_id}",
            text=subset["word"],
            textposition="top center",
            textfont=dict(size=11, family="Arial"),
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.82,
                line=dict(width=0.8, color="white"),
            ),
            customdata=subset[["word", "cluster", "freq"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Кластер: %{customdata[1]}<br>"
                "Частота: %{customdata[2]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text="Семантическая карта корпуса Заболоцкого<br>"
                 "<sup>индекс ДИКС · UMAP · K-Means</sup>",
            font=dict(size=16, color="#e0e0e0"),
            x=0.5,
        ),
        legend=dict(
            title="Кластеры",
            itemsizing="constant",
            bgcolor="rgba(45, 45, 45, 0.85)",
            bordercolor="#555555",
            borderwidth=1,
            font=dict(color="#e0e0e0"),
            title_font=dict(color="#e0e0e0"),
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#0d0d0d",
        margin=dict(l=20, r=20, t=80, b=20),
        height=600,
        hovermode="closest",
    )

    return fig

def format_context_with_highlight(text):
    """
    Преобразует маркеры <<<form>>> в красивую подсветку для Streamlit (:red[]).
    """
    text = re.sub(r'<<<([^>]+)>>>', r':red[\1]', text)
    return text

def display_contexts_table_simple(contexts):
    """
    Отображает контексты в виде обычной таблицы без выделения.
    Поддерживает встроенную сортировку Streamlit.
    """
    if not contexts:
        st.info("Контексты не найдены.")
        return

    contexts_df = pd.DataFrame(contexts)
    contexts_df = contexts_df.sort_values(by='Год')
    contexts_df.index = range(1, len(contexts_df) + 1)
    st.dataframe(contexts_df, width='stretch')

def display_contexts_table_highlighted(contexts):
    """
    Отображает контексты в виде markdown-таблицы с красивым выделением ключевого слова.
    """
    if not contexts:
        st.info("Контексты не найдены.")
        return

    # Сортируем по году
    sorted_contexts = sorted(contexts, key=lambda x: x['Год'])

    # Строим markdown таблицу
    markdown_table = "| № | Контекст | Произведение | Год |\n"
    markdown_table += "|---|----------|--------------|-----|\n"

    for i, ctx in enumerate(sorted_contexts, 1):
        # Форматируем контекст с выделением
        formatted_text = format_context_with_highlight(ctx['Контекст'])
        # Экранируем | в контексте и в названии произведения чтобы не сломалась таблица
        formatted_text = formatted_text.replace('|', '\\|')
        title_escaped = ctx['Произведение'].replace('|', '\\|')
        markdown_table += f"| {i} | {formatted_text} | {title_escaped} | {ctx['Год']} |\n"

    st.markdown(markdown_table)

@st.dialog("📊 Индекс контекстуальной близости — справка", width="large")
def show_diks_index_help():
    st.markdown("""
**ДИКС** — авторская динамическая метрика контекстуальной близости слов, учитывающая поэтическую структуру текстов (переносы строки, деление на строфы).
Логика индекса схожа с дистрибутивно-семантическим подходом.

---

#### 🆚 Чем отличается от классического контекста?

Классическое окно смотрит на *N* ближайших соседей вокруг слова.
индекс ДИКС сканирует **весь текст** каждого стихотворения и суммирует веса по **всем вхождениям** слова в корпусе, а также учитывает динамику - расстояние между словом и соседними словами - и структуру текста (переносы строк и границы предложений).

---

#### ⚙️ Формула веса связи

Для каждой пары (таргет - искомое слово, сосед - любая другая лексическая единица текста) вычисляется вес:
""")
    st.latex(r"\text{вес} = r^{d} \times b^{n\_brk} \times s^{n\_sent}")
    st.markdown("""
| Параметр | Что означает |
|---|---|
| **d** | расстояние в словах между таргетом и соседом |
| **n_brk** | число переносов строк (в том числе переносов "лесенкой") на пути |
| **n_sent** | число границ предложений на пути |
| **r, b, s** | коэффициенты затухания (настраиваются в ⚙️ Настройках) |
| |_Если в значении любого из параметров стоит единица - этот фактор не будет влиять на итоговый результат близости. Таким образом, можно производить тонкие подсчеты контекстуальных связей в соответствии с исследовательской задачей._|

Веса суммируются по всем вхождениям таргета — в результате получается **рейтинг** слов, наиболее тесно связанных с таргетом во всём корпусе.

---

#### Какой результат это даёт?

Индекс учитывает любые контекстуальные пересечения между словами и позволяет производить более точный и взвешенный подсчет контекстуальной близости слов в авторском корпусе.

""")

# --- Интерфейс Streamlit ---
st.set_page_config(page_title="Mayak-2D Prototype", layout="wide")

st.title("Zabolotsky Digital Corpus")
st.subheader("Цифровой корпус Н. А. Заболоцкого")

full_corpus = load_data()
lemmas_forms = load_lemma_forms()

@st.cache_data
def cached_full_word_analysis(search_word, year_range, window_size,
                               decay_distance, decay_brks, decay_sents):
    
    filtered = [
        item for item in full_corpus
        if year_range[0] <= item['year_finished'] <= year_range[1]
    ]

    return full_word_analysis(
        filtered_corpus=filtered,
        target_word=search_word,
        window_size=window_size,
        decay_distance=decay_distance,
        decay_brks=decay_brks,
        decay_sents=decay_sents,
        stopwords=russian_stopwords,
        lemma_forms=lemmas_forms
    )

# --- Боковая панель ---
search_word = st.sidebar.text_input("Введите слово для анализа", "лошадь")
window_size = st.sidebar.slider("Размер окна контекста", 1, 15, 7)

count_stopwords = st.sidebar.checkbox('Учитывать служебные слова', value=False)

if full_corpus:
    min_year = min(item['year_finished'] for item in full_corpus)
    max_year = max(item['year_finished'] for item in full_corpus)
else:
    min_year = max_year = 0
    st.error("Корпус не загружен. Проверьте файл data/database.parquet.")

year_range = st.sidebar.slider(
    "Период написания",
    min_year, max_year, (min_year, max_year)
)

compare_periods = st.sidebar.checkbox("Добавить второй период для сравнения контекстов", value=False)

if compare_periods:
    year_range_2 = st.sidebar.slider(
        "Период написания (для сравнения)",
        min_year, max_year, (min_year, max_year)
    )

with st.sidebar.expander("👾 Настройки LLM"):
    model_source = st.radio(
    "Модель анализа:",
    ["Локальная (Ollama)", "DeepSeek 4 (API)"],
    index=1,
    help="Ollama требует скачивания модели локально. DeepSeek отправляет запрос через интернет."
    )

with st.sidebar.expander("⚙️ Настройки весов (индекс ДИКС)"):
    decay_distance = st.slider(
        "Затухание от расстояния",
        min_value=0.5, max_value=1.0, value=0.95, step=0.01,
        help="Коэффициент затухания для слов, находящихся дальше от таргета."
    )
    decay_sents = st.slider(
        "Между предложениями",
        min_value=0.1, max_value=1.0, value=0.9, step=0.05,
        help="Коэффициент затухания при переходе к следующему предложению."
    )
    decay_brks = st.slider(
        "Между разрывами строки (_BRK_)",
        min_value=0.1, max_value=1.0, value=0.85, step=0.05,
        help="Коэффициент затухания за перенос строки или 'лесенку'."
    )

# --- Справка об индексе ДИКС ---
st.sidebar.divider()
if st.sidebar.button("ℹ️ Что такое индекс ДИКС?", use_container_width=True):
    show_diks_index_help()

# --- Глобальные вкладки ---
tab_search, tab_corpus, tab_lexical_comparison = st.tabs(["🔍 Анализ слова", "📊 Статистика корпуса", "📝 Лексические пласты"])

# ══════════════════════════════════════════════════════════════
# ВКЛАДКА 1: Анализ слова
# ══════════════════════════════════════════════════════════════
with tab_search:

    if search_word:

        search_word = search_word.strip().lower().replace('ё', 'е')

        found_lemma = next(
            (lemma for lemma in lemmas_forms if lemma.replace('ё', 'е') == search_word),
            None
        )
        if not found_lemma:
            found_lemma = next(
                (lemma for lemma, forms in lemmas_forms.items()
                 if any(f.replace('ё', 'е') == search_word for f in forms)),
                None
            )
        if found_lemma:
            target_word = found_lemma
            search_word = found_lemma
        else:
            st.warning("Слово не найдено в корпусе.")
            st.stop()

        filtered_corpus = [
            item for item in full_corpus
            if year_range[0] <= item['year_finished'] <= year_range[1]
        ]

        results = cached_full_word_analysis(
            search_word, year_range, window_size,
            decay_distance, decay_brks, decay_sents
        )

        # Анализ второго периода для сравнения (если включено)
        results_2 = None
        if compare_periods:
            if year_range == year_range_2:
                results_2 = results
            else:
                results_2 = cached_full_word_analysis(
                    search_word, year_range_2, window_size,
                    decay_distance, decay_brks, decay_sents
                )

        if not results:
            st.warning("Слово не найдено в корпусе.")

        else:
            # Извлекаем данные для удобства
            total_occurrences = results['total_occurrences']
            contexts = results['contexts']
            year_dist = results['year_dist']
            top_neighbors = results['window_neighbors']
            pos_dist = results['pos_dist']
            proximity_weights = results['proximity_weights']

            # --- УРОВЕНЬ 1.1: Заголовок со словом ---
            st.markdown(f"## Анализ слова: `{target_word}`")
            if compare_periods:
                st.caption(f"📊 Сравнение периодов: {year_range[0]} — {year_range[1]} vs {year_range_2[0]} — {year_range_2[1]}")
            else:
                st.caption(f"Период поиска: {year_range[0]} — {year_range[1]}")

            # --- УРОВЕНЬ 1.2: Синонимы (из словаря и встречающиеся в корпусе) ---
            st.subheader("Семантический кластер")
            synonyms = cached_get_unique_synonyms(search_word, top_n=20, depth=50)
            synonyms_filtered = filter_synonyms_by_corpus(synonyms)

            if synonyms:
                if compare_periods:
                    st.info("Подсчет семантически близких слов (векторных синонимов) происходит без привязки к периоду (на основе общего векторного словаря и полного корпуса Заболоцкого)")
                show_coefficients = st.checkbox('Показать коэффициенты близости', value=True)
                if show_coefficients:
                    synonyms_str = ', '.join([f"{syn} ({score:.4f})" for syn, score in synonyms])
                    st.write(f"Семантически близкие слова по общему корпусу художественной литературы (с коэффициентами близости): {synonyms_str}")
                else:
                    st.write(f"Семантически близкие слова по общему корпусу художественной литературы (без коэффициентов): {', '.join([syn for syn, score in synonyms])}")
                st.info("Список включает слова из общего векторного словаря, которые могут не встречаться в поэтических текстах Заболоцкого. Ниже — только те слова, которые действительно есть в корпусе")
                st.write(f"Семантически близкие слова, найденные в корпусе: {', '.join(synonyms_filtered)}")
            else:
                st.write("Семантически близкие слова не найдены или слово отсутствует в модели.")

            st.divider()

            # --- УРОВЕНЬ 2: Метрики и графики (один период или сравнение) ---
            if compare_periods and results_2:
                # Извлекаем данные для обоих периодов
                total_occurrences_2 = results_2['total_occurrences']
                contexts_2 = results_2['contexts']
                year_dist_2 = results_2['year_dist']
                pos_dist_2 = results_2['pos_dist']

                # === ПЕРИОД 1 ===
                st.subheader(f"📍 Период {year_range[0]} — {year_range[1]}")
                col1_metric, col1_pos, col1_years = st.columns([1, 1.2, 1.2])

                with col1_metric:
                    st.metric("Всего употреблений", total_occurrences)
                    total_texts_1 = len(set(ctx.get('Произведение', '') for ctx in contexts if ctx.get('Произведение')))
                    st.metric("Всего текстов с леммой:", total_texts_1)

                with col1_pos:
                    st.caption("Частеречное окружение")
                    if count_stopwords:
                        pos_data = pos_dist['with_stopwords']
                    else:
                        pos_data = pos_dist['filtered']
                    pos_df = pd.DataFrame(pos_data.items(), columns=['Часть речи', 'Кол-во'])
                    st.bar_chart(pos_df.set_index('Часть речи'), height=200)

                with col1_years:
                    st.caption("Динамика")
                    year_df = pd.DataFrame(year_dist.items(), columns=['Год', 'Частота']).sort_values('Год')
                    st.line_chart(year_df.set_index('Год'), height=200)

                # === ПЕРИОД 2 ===
                st.subheader(f"📍 Период {year_range_2[0]} — {year_range_2[1]}")
                col2_metric, col2_pos, col2_years = st.columns([1, 1.2, 1.2])

                with col2_metric:
                    st.metric("Всего употреблений", total_occurrences_2)
                    total_texts_2 = len(set(ctx.get('Произведение', '') for ctx in contexts_2 if ctx.get('Произведение')))
                    st.metric("Всего текстов с леммой:", total_texts_2)

                with col2_pos:
                    st.caption("Частеречное окружение")
                    if count_stopwords:
                        pos_data_2 = pos_dist_2['with_stopwords']
                    else:
                        pos_data_2 = pos_dist_2['filtered']
                    pos_df_2 = pd.DataFrame(pos_data_2.items(), columns=['Часть речи', 'Кол-во'])
                    st.bar_chart(pos_df_2.set_index('Часть речи'), height=200)

                with col2_years:
                    st.caption("Динамика")
                    year_df_2 = pd.DataFrame(year_dist_2.items(), columns=['Год', 'Частота']).sort_values('Год')
                    st.line_chart(year_df_2.set_index('Год'), height=200)

                # Дельта-метрики
                st.divider()
                occ_delta = total_occurrences_2 - total_occurrences
                occ_pct = occ_delta / max(total_occurrences, 1) * 100
                col_delta, _, _ = st.columns(3)
                with col_delta:
                    st.metric("Δ Употреблений", f"{occ_delta:+d}", f"{occ_pct:+.1f}%")
            else:
                # Режим одного периода: три колонки как было
                col_metric, col_pos, col_years = st.columns(3)

                with col_metric:
                    st.subheader("Статистика")
                    st.metric("Всего употреблений", total_occurrences)
                    total_texts = len(set(ctx.get('Произведение', '') for ctx in contexts if ctx.get('Произведение')))
                    st.metric("Всего текстов с леммой:", total_texts)

                with col_pos:
                    st.subheader("Частеречное окружение")
                    if count_stopwords:
                        pos_data = pos_dist['with_stopwords']
                    else:
                        pos_data = pos_dist['filtered']
                    pos_df = pd.DataFrame(pos_data.items(), columns=['Часть речи', 'Кол-во'])
                    st.bar_chart(pos_df.set_index('Часть речи'))

                with col_years:
                    st.subheader("Динамика")
                    year_df = pd.DataFrame(results['year_dist'].items(), columns=['Год', 'Частота']).sort_values('Год')
                    st.line_chart(year_df.set_index('Год'))

            st.divider()

            # --- УРОВЕНЬ 3: Сравнение методов (на всю ширину) ---
            st.subheader("Семантические связи")

            if compare_periods and results_2:
                
                top_neighbors_2 = results_2['window_neighbors']
                proximity_weights_2 = results_2['proximity_weights']

                delta_analysis = calculate_delta_analysis(results, results_2, count_stopwords=count_stopwords)

                tab_window, tab_index, tab_delta = st.tabs(["Классическое окно", "индекс ДИКС", "Дельта-анализ"])

                with tab_window:
                    # Классическое окно для обоих периодов
                    col_wnd_1, col_wnd_2 = st.columns(2)

                    with col_wnd_1:
                        st.caption(f"Период {year_range[0]}—{year_range[1]}")
                        if count_stopwords:
                            n_df = pd.DataFrame(top_neighbors['with_stopwords'].most_common(10), columns=['Лемма', 'Частота'])
                        else:
                            n_df = pd.DataFrame(top_neighbors['filtered'].most_common(10), columns=['Лемма', 'Частота'])
                        n_df.index = range(1, len(n_df) + 1)
                        st.table(n_df)

                    with col_wnd_2:
                        st.caption(f"Период {year_range_2[0]}—{year_range_2[1]}")
                        if count_stopwords:
                            n_df_2 = pd.DataFrame(top_neighbors_2['with_stopwords'].most_common(10), columns=['Лемма', 'Частота'])
                        else:
                            n_df_2 = pd.DataFrame(top_neighbors_2['filtered'].most_common(10), columns=['Лемма', 'Частота'])
                        n_df_2.index = range(1, len(n_df_2) + 1)
                        st.table(n_df_2)

                with tab_index:
                    # Индекс контекстуальной близости для обоих периодов
                    col_idx_1, col_idx_2 = st.columns(2)

                    with col_idx_1:
                        st.caption(f"Период {year_range[0]}—{year_range[1]}")
                        weights_df = pd.DataFrame(proximity_weights.most_common(10), columns=['Лемма', 'Индекс'])
                        if not weights_df.empty:
                            max_val = weights_df['Индекс'].max()
                            weights_df['Сила связи'] = weights_df['Индекс'] / max_val
                            weights_df.index = range(1, len(weights_df) + 1)
                            st.dataframe(
                                weights_df[['Лемма', 'Сила связи']],
                                column_config={
                                    "Сила связи": st.column_config.ProgressColumn(
                                        "Контекстуальная близость", format="%.2f", min_value=0, max_value=1
                                    )
                                },
                                width='stretch'
                            )

                    with col_idx_2:
                        st.caption(f"Период {year_range_2[0]}—{year_range_2[1]}")
                        weights_df_2 = pd.DataFrame(proximity_weights_2.most_common(10), columns=['Лемма', 'Индекс'])
                        if not weights_df_2.empty:
                            max_val_2 = weights_df_2['Индекс'].max()
                            weights_df_2['Сила связи'] = weights_df_2['Индекс'] / max_val_2
                            weights_df_2.index = range(1, len(weights_df_2) + 1)
                            st.dataframe(
                                weights_df_2[['Лемма', 'Сила связи']],
                                column_config={
                                    "Сила связи": st.column_config.ProgressColumn(
                                        "Контекстуальная близость", format="%.2f", min_value=0, max_value=1
                                    )
                                },
                                width='stretch'
                            )

                with tab_delta:
                    st.markdown("### 📈 Анализ изменений семантического поля")

                    if delta_analysis is None:
                        st.warning("Нет данных для дельта-анализа.")
                    else:
                        # Появившиеся слова
                        col_app, col_dis = st.columns(2)

                        with col_app:
                            st.subheader("🟢 Топ появившихся слов")
                            if delta_analysis['appeared_words']:
                                app_df = pd.DataFrame(
                                    delta_analysis['appeared_words'],
                                    columns=['Слово', 'Индекс']
                                )
                                app_df.index = range(1, len(app_df) + 1)
                                st.dataframe(app_df.head(10), width='stretch')
                            else:
                                st.info("Нет новых слов.")

                        with col_dis:
                            st.subheader("🔴 Топ исчезнувших слов")
                            if delta_analysis['disappeared_words']:
                                dis_df = pd.DataFrame(
                                    delta_analysis['disappeared_words'],
                                    columns=['Слово', 'Индекс']
                                )
                                dis_df.index = range(1, len(dis_df) + 1)
                                st.dataframe(dis_df.head(10), width='stretch')
                            else:
                                st.info("Нет исчезнувших слов.")

                        st.divider()

                        # Изменяющиеся слова
                        st.subheader("🔄 Самые существенные изменения индекса контекстуальной близости")

                        if delta_analysis['changed_words']:
                            changed_viz_data = []
                            for item in delta_analysis['changed_words'][:10]:
                                changed_viz_data.append({
                                    'Слово': item['word'],
                                    'Индекс период 1': f"{item['index_1']:.3f}",
                                    'Индекс период 2': f"{item['index_2']:.3f}",
                                    'Δ Индекс': f"{item['index_delta']:+.3f}",
                                    'Δ %': f"{item['index_pct']:+.1f}%",
                                    'Статус': '📈' if item['status'] == 'growing' else ('📉' if item['status'] == 'declining' else '➡️')
                                })

                            changed_df = pd.DataFrame(changed_viz_data)
                            changed_df.index = range(1, len(changed_df) + 1)
                            st.dataframe(changed_df, width='stretch', hide_index=False)
                        else:
                            st.info("Нет изменяющихся слов.")
            else:
                # Режим одного периода
                tab_window, tab_index = st.tabs(["Классическое окно контекста", "индекс ДИКС"])

                with tab_window:
                    if count_stopwords:
                        n_df = pd.DataFrame(top_neighbors['with_stopwords'].most_common(10), columns=['Лемма', 'Частота'])
                    else:
                        n_df = pd.DataFrame(top_neighbors['filtered'].most_common(10), columns=['Лемма', 'Частота'])
                    n_df.index = range(1, len(n_df) + 1)
                    st.table(n_df)

                with tab_index:
                    weights_df = pd.DataFrame(proximity_weights.most_common(10), columns=['Лемма', 'Индекс'])

                    if not weights_df.empty:
                        max_val = weights_df['Индекс'].max()
                        weights_df['Сила связи'] = weights_df['Индекс'] / max_val
                        weights_df.index = range(1, len(weights_df) + 1)

                        st.dataframe(
                            weights_df[['Лемма', 'Сила связи']],
                            column_config={
                                "Сила связи": st.column_config.ProgressColumn(
                                    "Контекстуальная близость", format="%.2f", min_value=0, max_value=1
                                )
                            },
                            width='stretch'
                        )

            # Таблица контекстов
            st.write("### Контексты употребления")

            if compare_periods and results_2:
                # Режим сравнения: контексты для обоих периодов
                contexts_2 = results_2['contexts']

                col_ctx_1, col_ctx_2 = st.columns(2)

                with col_ctx_1:
                    st.subheader(f"Период {year_range[0]} — {year_range[1]} ({len(contexts)} контекстов)")
                    if contexts:
                        context_format = st.radio(
                            "Формат отображения (период 1):",
                            ["📝 Таблица (базовая)", "✍️ Таблица (с выделением)"],
                            horizontal=True,
                            help="Выберите удобный способ просмотра контекстов",
                            key="ctx_fmt_1"
                        )
                        if context_format == "📝 Таблица (базовая)":
                            display_contexts_table_simple(contexts)
                        else:
                            display_contexts_table_highlighted(contexts)

                with col_ctx_2:
                    st.subheader(f"Период {year_range_2[0]} — {year_range_2[1]} ({len(contexts_2)} контекстов)")
                    if contexts_2:
                        context_format_2 = st.radio(
                            "Формат отображения (период 2):",
                            ["📝 Таблица (базовая)", "✍️ Таблица (с выделением)"],
                            horizontal=True,
                            help="Выберите удобный способ просмотра контекстов",
                            key="ctx_fmt_2"
                        )
                        if context_format_2 == "📝 Таблица (базовая)":
                            display_contexts_table_simple(contexts_2)
                        else:
                            display_contexts_table_highlighted(contexts_2)
                    else:
                        st.info("Контексты не найдены в этом периоде.")
            else:
                # Режим одного периода
                if contexts:
                    context_format = st.radio(
                        "Формат отображения:",
                        ["📝 Таблица (базовая)", "✍️ Таблица (с выделением)"],
                        horizontal=True,
                        help="Выберите удобный способ просмотра контекстов"
                    )

                    if context_format == "📝 Таблица (базовая)":
                        display_contexts_table_simple(contexts)
                    else:
                        display_contexts_table_highlighted(contexts)

    # --- БЛОК ИНТЕРПРЕТАЦИИ ЧЕРЕЗ LLM ---
    if search_word and results:
        if st.button("🚀 Запустить анализ через LLM"):

            status_text = st.empty()

            with st.spinner("Собираем статистику для промпта... Пожалуйста, подождите."):

                # Считаем близость синонимов к таргету в этом периоде
                status_text.text("📊 Рассчитываем семантическую близость синонимов...")
                syn_prox_index = synonyms_proximity_index(target_word, synonyms_filtered, results['proximity_weights'])

                # Считаем контекстуальные связи для каждого синонима
                status_text.text("🤓 Считаем индекс ДИКС для синонимов (это может занять время)...")
                neighbors_for_syns = proximity_neighbours_for_synonyms(
                    synonyms_filtered,
                    filtered_corpus,
                    decay_distance, decay_brks, decay_sents,
                    stopwords=russian_stopwords
                )

                # Сборка промпта
                status_text.text("✍️ Формирую аналитическое досье для ИИ...")
                interpr_prompt = prepare_llm_prompt(
                    target_word=target_word,
                    synonyms=synonyms,
                    synonyms_filtered=synonyms_filtered,
                    syn_proximity=syn_prox_index,
                    neighbors_for_synonyms=neighbors_for_syns,
                    total_occurrences=total_occurrences,
                    year_dist=year_dist,
                    proximity_weights=proximity_weights
                )

                # Убираем временный текст статуса перед выводом результата
                status_text.empty()

            # Наглядный вывод промпта для проверки
            st.subheader("Сгенерированный промпт для ИИ:")
            st.code(interpr_prompt, language="text")

            st.divider()
            st.subheader("📝 Аналитический комментарий от LLM:")

            st.info("Несмотря на предварительную настройку, LLM может добавлять к реальным данным собственные интерпретации. Пожалуйста, относитесь к результату критически и сверяйтесь с фактическими данными из предыдущих разделов.")

            # --- ЛОГИКА ВЫБОРА МОДЕЛИ ---

            if model_source == "Локальная (Ollama)":

                try:
                    import ollama
                except ImportError:
                    st.error("Модуль ollama не установлен. Пожалуйста, установите его с помощью команды: pip install ollama")

                response_container = st.empty()
                ollama_prompt = prompt_prefix + "\n\n" + interpr_prompt
                full_response = ""

                try:
                    stream = ollama.generate(model='llama3:8b', prompt=ollama_prompt, stream=True)
                    for chunk in stream:
                        full_response += chunk['response']
                        response_container.markdown(full_response + "▌")
                    response_container.markdown(full_response)

                except Exception as e:
                    st.error(f"Ошибка при обращении к Ollama: {e}")
                    st.info("Убедитесь, что приложение Ollama запущено и модель llama3:8b скачана.")

            elif model_source == "DeepSeek 4 (API)":
                if not deepseek_key:
                    st.error("Ключ DeepSeek не найден в .env! Добавьте DEEPSEEK_API_KEY.")
                else:

                    from openai import OpenAI as DeepSeekClient
                    
                    client_ds = DeepSeekClient(api_key=deepseek_key, base_url="https://api.deepseek.com")

                    with st.spinner("DeepSeek анализирует семантические поля..."):
                        try:
                            response = client_ds.chat.completions.create(
                                model="deepseek-v4-flash",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": prompt_prefix
                                    },
                                    {
                                        "role": "user",
                                        "content": interpr_prompt,
                                    }
                                ],
                                stream=False,
                                extra_body={"thinking": {"type": "enabled"}}
                            )
                            st.markdown(response.choices[0].message.content)
                        except Exception as e:
                            st.error(f"Ошибка API DeepSeek: {e}")
                            st.info("Убедитесь, что ключ DEEPSEEK_API_KEY корректно настроен.")


# ══════════════════════════════════════════════════════════════
# ВКЛАДКА 2: Статистика корпуса
# ══════════════════════════════════════════════════════════════
with tab_corpus:
    st.markdown("## Статистика корпуса")

    # Фильтр по периоду внутри вкладки
    stats_year_range = st.slider(
        "Период",
        min_year, max_year, (min_year, max_year),
        key="stats_year_range"
    )

    filtered_corpus_stats = [
        item for item in full_corpus
        if stats_year_range[0] <= item['year_finished'] <= stats_year_range[1]
    ]

    if not filtered_corpus_stats:
        st.warning("Нет данных за выбранный период.")
    else:
        tab_corp_metrics, tab_corp_freq, tab_corp_growth = st.tabs([
            "📋 Общие метрики",
            "📖 Частотный словарь",
            "📈 Рост словаря",
        ])

        # --- 2.1: Общие метрики ---
        with tab_corp_metrics:
            metrics = compute_general_metrics(filtered_corpus_stats)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Произведений", metrics['total_texts'])
            col2.metric("Предложений", f"{metrics['total_tokens']:,}".replace(',', '\u00a0'))
            col3.metric("Лемм всего", f"{metrics['total_lemmas']:,}".replace(',', '\u00a0'))
            col4.metric("Уникальных лемм", f"{metrics['unique_lemmas']:,}".replace(',', '\u00a0'))

            st.divider()
            st.subheader("Тексты по годам")
            year_texts_df = pd.DataFrame(
                sorted(metrics['texts_by_year'].items()),
                columns=['Год', 'Текстов']
            ).set_index('Год')
            st.bar_chart(year_texts_df)

        # --- 2.2: Частотный словарь ---
        with tab_corp_freq:
            col_sw, col_n = st.columns([1, 2])
            with col_sw:
                exclude_sw = st.checkbox("Исключить стоп-слова", value=True, key="freq_exclude_sw")
            with col_n:
                top_n = st.slider("Топ-N слов", 10, 300, 50, key="freq_top_n")

            freq_counter = compute_frequency_dict(
                filtered_corpus_stats,
                exclude_stopwords=exclude_sw
            )
            total_lemmas_count = sum(freq_counter.values())
            top_lemmas = freq_counter.most_common(top_n)

            freq_df = pd.DataFrame(top_lemmas, columns=['Лемма', 'Частота'])
            freq_df['% от корпуса'] = (freq_df['Частота'] / total_lemmas_count * 100).round(3)
            freq_df.index = range(1, len(freq_df) + 1)

            col_table, col_chart = st.columns([1, 1.2])
            with col_table:
                st.dataframe(freq_df, width='stretch')
            with col_chart:
                st.bar_chart(freq_df.set_index('Лемма')['Частота'].head(30))

        # --- 2.3: Рост словарного запаса ---
        with tab_corp_growth:
            growth_data = compute_vocabulary_growth(filtered_corpus_stats)
            growth_df = pd.DataFrame(growth_data).set_index('Год')

            if not growth_df.empty:
                st.subheader("Накопленный словарный запас")
                st.caption("Сколько уникальных лемм встречено в корпусе к каждому году")
                st.line_chart(growth_df[['Уникальных лемм накоплено']])

                st.divider()
                st.subheader("Новых уникальных лемм в год")
                st.caption("Сколько ранее не встречавшихся лемм появилось в текстах каждого года")
                st.bar_chart(growth_df[['Новых лемм']])

                st.divider()
                st.subheader("Type-Token Ratio")
                st.caption("Отношение уникальных лемм к общему числу лемм в году")
                st.line_chart(growth_df[['Type-Token Ratio']])

        st.divider()

        st.title("Векторная карта самых частотных слов")
        st.info("Основана на векторных представлениях слов из модели Navec, обученной на корпусе русской литературы.")

        col_vm_sw, col_vm_n = st.columns([1, 2])
        with col_vm_sw:
            vm_exclude_sw = st.checkbox("Исключить стоп-слова", value=True, key="vm_exclude_sw")
        with col_vm_n:
            vm_top_n = st.slider("Количество слов", 20, 300, 100, key="vm_top_n")

        map_df = compute_vector_map(filtered_corpus_stats, top_n=vm_top_n, exclude_stopwords=vm_exclude_sw, pca_base_size=max(300, vm_top_n))

        if map_df is None:
            st.warning("Недостаточно слов с векторными представлениями для построения карты.")
        else:
            points = alt.Chart(map_df).mark_circle(opacity=0.7).encode(
                x=alt.X('x:Q', axis=alt.Axis(labels=False, ticks=False, title=None, grid=False)),
                y=alt.Y('y:Q', axis=alt.Axis(labels=False, ticks=False, title=None, grid=False)),
                size=alt.Size('Частота:Q', scale=alt.Scale(range=[40, 400]), legend=None),
                color=alt.Color('Частота:Q', scale=alt.Scale(scheme='viridis'), legend=None),
                tooltip=['Слово:N', 'Частота:Q'],
            )
            labels = alt.Chart(map_df).mark_text(dx=6, dy=-6, fontSize=11, align='left', color='white').encode(
                x='x:Q',
                y='y:Q',
                text='Слово:N',
                tooltip=['Слово:N', 'Частота:Q'],
            )
            st.altair_chart(
                (points + labels).properties(height=600).configure_view(strokeWidth=0),
                width='stretch',
            )

        st.divider()

        st.title("Семантическая карта по Индексу ДИКС")

        # Загружаем карту кластеров
        cluster_map_df = load_cluster_map()

        if cluster_map_df is None:
            st.info(
                "📊 Карта контекстуальных кластеров ещё не сгенерирована. "
                "Её нужно предрассчитать из корня проекта:\n\n"
                "```bash\npython -m src.map_builder\n```\n\n"
                "Это займёт 20–30 минут в первый раз (300 слов × 20 кластеров)."
            )
        else:
            # Контролы для семантической карты: поиск + разделитель + чекбокс размер
            col_search, col_sep, col_size = st.columns([0.8, 0.02, 0.18], gap="small")

            with col_search:
                sm_search_word = st.text_input(
                    "🔍 Найти слово:",
                    placeholder="Например: революция, любовь, рабочий...",
                    key="semantic_map_search"
                )

            # col_sep — пустая колонка для визуального разделителя

            with col_size:
                sm_size_by_freq = st.checkbox(
                    "Размер ~ частота",
                    value=True,
                    key="semantic_map_size_freq",
                    help="Крупнее = чаще встречается"
                )

            search_result = None
            if sm_search_word:
                match = cluster_map_df[cluster_map_df["word"] == sm_search_word.strip().lower()]
                if not match.empty:
                    search_result = match.iloc[0]

            if sm_search_word:
                if search_result is None:
                    st.info(f"Слово «{sm_search_word}» не входит в топ-300 слов карты.")
                else:
                    st.success(
                        f"**{search_result['word']}** — Кластер {int(search_result['cluster'])}, "
                        f"частота: {int(search_result['freq'])}, "
                        f"координаты: ({search_result['x']:.2f}, {search_result['y']:.2f})"
                    )

            all_cluster_ids = sorted(cluster_map_df["cluster"].unique().tolist())
            selected_clusters = all_cluster_ids

            fig = build_mayak_semantic_map(cluster_map_df, selected_clusters, sm_size_by_freq)
            st.plotly_chart(fig, width='stretch')

            with st.expander("📋 Состав кластеров"):
                display_df = (
                    cluster_map_df[cluster_map_df["cluster"].isin(selected_clusters)]
                    [["cluster", "word", "freq"]]
                    .rename(columns={"cluster": "Кластер", "word": "Слово", "freq": "Частота"})
                    .sort_values(["Кластер", "Частота"], ascending=[True, False])
                    .reset_index(drop=True)
                )
                # Добавляем номер строки начиная с 1
                display_df.index = display_df.index + 1
                st.dataframe(display_df, width='stretch')


# ══════════════════════════════════════════════════════════════
# ВКЛАДКА 3: Лексические пласты
# ══════════════════════════════════════════════════════════════
with tab_lexical_comparison:

    st.markdown("## 📝 Лексические пласты Заболоцкого")

    if isinstance(full_corpus, pd.DataFrame):
        df = full_corpus.copy()
    else:
        # Если пришел список словарей (list), списки объектов или что-то еще
        df = pd.DataFrame(list(full_corpus))
        
    # Сбрасываем индексы, чтобы избежать внутренних конфликтов масок в pandas
    df = df.reset_index(drop=True)

    df['year_int'] = pd.to_numeric(df['year_finished'], errors='coerce')
    min_available_year = int(df['year_int'].min()) if not df['year_int'].isnull().all() else 1920
    max_available_year = int(df['year_int'].max()) if not df['year_int'].isnull().all() else 1958

    st.markdown("""
    Эта вкладка позволяет сравнить лексику Заболоцкого между двумя любыми периодами его творчества. 
    Метод находит слова, которые в **Исследуемом периоде** используются значимо чаще, чем в **Референсном**.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Исследуемый период (Target)")
        target_range = st.slider("Выберите диапазон годов", min_available_year, max_available_year, (1946, max_available_year), key="t_slide")

    with col2:
        st.subheader("📚 Референсный период (Reference)")
        ref_range = st.slider("Выберите диапазон годов для сравнения", min_available_year, max_available_year, (min_available_year, 1933), key="r_slide")

    top_k = st.number_input("Количество результатов", min_value=5, max_value=100, value=25)

    if st.button("🚀 Запустить корпусный анализ", type="primary"):
        if target_range[0] == ref_range[0] and target_range[1] == ref_range[1]:
            st.warning("⚠️ Периоды полностью совпадают. Сравнение не имеет математического смысла.")
        else:
            with st.spinner("Рассчитываем метрику логарифмического правдоподобия..."):
                keywords, total_t, total_r = calculate_diachronic_log_likelihood(
                    df, target_range, ref_range, top_n=top_k
                )
                label_col = "Лексема (Маркер)"
                file_prefix = "words"
                
            if not keywords:
                st.error("В выбранных диапазонах годов не найдено достаточно данных.")
            else:
                m_col1, m_col2 = st.columns(2)
                m_col1.metric(f"Всего слов в исследуемом корпусе", f"{total_t:,}")
                m_col2.metric(f"Всего слов в референсном корпусе", f"{total_r:,}")
                
                st.success(f"Анализ завершен! Найдено {len(keywords)} лексем (p < 0.05).")
                
                res_data = []
                for rank, (item, score) in enumerate(keywords, 1):
                    res_data.append({
                        "Ранг": rank,
                        label_col: item,
                        "Сила связи (Log-Likelihood)": round(score, 2),
                        "Значимость": "Высокая (p < 0.01)" if score > 6.63 else "Стандартная (p < 0.05)"
                    })
                    
                res_df = pd.DataFrame(res_data)
                st.dataframe(res_df.set_index("Ранг"), use_container_width=True)
                
                csv_buffer = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать результаты (.CSV)",
                    data=csv_buffer,
                    file_name=f"zabolotsky_{file_prefix}_{target_range[0]}-{target_range[1]}_vs_{ref_range[0]}-{ref_range[1]}.csv",
                    mime="text/csv"
                )

    st.markdown('## Сравнение семантических карт')

    # Синхронизированный поиск: ввод в этом поле подсвечивает слово на обеих картах
    sync_search = st.text_input("🔎 Синхронизированный поиск (подсветка в обеих картах)", key="sync_map_search")

    # Вертикальное расположение карт (одна под другой) для удобства масштабирования
    st.subheader("Карта — период 1918–1938")
    cluster_map_1 = load_cluster_map_first_period()
    if cluster_map_1 is None:
        st.info(
            "Карта для периода 1918–1938 не найдена. Сгенерируйте её через:\n" \
            "`python -m src.map_builder --period 1918-1938 --output-path data/word_map_1918_1938.json`"
        )
    else:
        with st.expander("Параметры карты (период 1)"):
            sm_size_by_freq_1 = st.checkbox("Размер ~ частота (период 1)", value=True, key="map1_size")

            cluster_palette = [
                "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
                "#9B5DE5", "#00BBF9", "#F15BB5", "#06D6A0", "#FB5607",
                "#8338EC", "#3A86FF", "#FFBE0B", "#FF006E", "#8AC926",
                "#DC2F02", "#370617", "#03071E", "#D62828", "#F77F00",
            ]

            cluster_rows = []
            for cluster_id in sorted(cluster_map_1["cluster"].unique()):
                color = cluster_palette[(cluster_id - 1) % len(cluster_palette)]
                lemmas = sorted(cluster_map_1[cluster_map_1["cluster"] == cluster_id]["word"].tolist())
                cluster_rows.append({
                    "Кластер": int(cluster_id),
                    "Цвет": color,
                    "Леммы": ", ".join(lemmas)
                })

            st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True)

        highlight_word_1 = sync_search.strip().lower()

        fig1 = build_mayak_semantic_map(cluster_map_1, sorted(cluster_map_1["cluster"].unique().tolist()), sm_size_by_freq_1)

        # Добавляем подсветку слова, если задано
        if highlight_word_1:
            match = cluster_map_1[cluster_map_1["word"] == highlight_word_1]
            if not match.empty:
                r = match.iloc[0]
                fig1.add_trace(go.Scatter(
                    x=[r['x']], y=[r['y']], mode='markers+text', text=[r['word']], textposition='top center',
                    marker=dict(size=30, color='#FFFF00', symbol='star', line=dict(width=1, color='#333333')),
                    showlegend=False, hoverinfo='text', name='highlight'
                ))
                st.success(f"**{r['word']}** — Кластер {int(r['cluster'])}, частота: {int(r['freq'])}")

        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    st.subheader("Карта — период 1946–1958")
    cluster_map_2 = load_cluster_map_last_period()
    if cluster_map_2 is None:
        st.info(
            "Карта для периода 1946–1958 не найдена. Сгенерируйте её через:\n" \
            "`python -m src.map_builder --period 1946-1958 --output-path data/word_map_1946_1958.json`"
        )
    else:
        with st.expander("Параметры карты (период 2)"):
            sm_size_by_freq_2 = st.checkbox("Размер ~ частота (период 2)", value=True, key="map2_size")

            cluster_palette = [
                "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
                "#9B5DE5", "#00BBF9", "#F15BB5", "#06D6A0", "#FB5607",
                "#8338EC", "#3A86FF", "#FFBE0B", "#FF006E", "#8AC926",
                "#DC2F02", "#370617", "#03071E", "#D62828", "#F77F00",
            ]

            cluster_rows = []
            for cluster_id in sorted(cluster_map_2["cluster"].unique()):
                color = cluster_palette[(cluster_id - 1) % len(cluster_palette)]
                lemmas = sorted(cluster_map_2[cluster_map_2["cluster"] == cluster_id]["word"].tolist())
                cluster_rows.append({
                    "Кластер": int(cluster_id),
                    "Цвет": color,
                    "Леммы": ", ".join(lemmas)
                })

            st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True)

        highlight_word_2 = sync_search.strip().lower()

        fig2 = build_mayak_semantic_map(cluster_map_2, sorted(cluster_map_2["cluster"].unique().tolist()), sm_size_by_freq_2)

        if highlight_word_2:
            match = cluster_map_2[cluster_map_2["word"] == highlight_word_2]
            if not match.empty:
                r = match.iloc[0]
                fig2.add_trace(go.Scatter(
                    x=[r['x']], y=[r['y']], mode='markers+text', text=[r['word']], textposition='top center',
                    marker=dict(size=30, color='#FFFF00', symbol='star', line=dict(width=1, color='#333333')),
                    showlegend=False, hoverinfo='text', name='highlight'
                ))
                st.success(f"**{r['word']}** — Кластер {int(r['cluster'])}, частота: {int(r['freq'])}")

        st.plotly_chart(fig2, use_container_width=True)

    st.link_button("Посмотреть подготовленную интерпретацию карт от LLM", cluster_maps_interpr_link)

    st.info(
        "⚠️ **BETA версия**: Функционал анализа неологизмов находится в стадии разработки. "
        "В данный момент корректная фильтрация результатов ещё не реализована. "
        "Список будет дополняться дополнительными фильтрами и возможностями анализа."
    )

    # Загружаем гапаксы
    hapax_data = load_mayak_hapax()

    if hapax_data is None:
        st.warning(
            "❌ Файл с данными гапаксов не найден. "
        )
    else:
        hapax_metadata = hapax_data.get('metadata', {})
        hapax_set = set(hapax_data.get('hapax_legomena', []))

        # Показываем статистику
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Произведений Заболоцкого",
            hapax_metadata.get('zabolotsky_poems_count', 0)
        )
        col2.metric(
            "Уникальных лемм (Заболоцкий)",
            f"{hapax_metadata.get('zabolotsky_unique_lemmas', 0):,}".replace(',', ' ')
        )
        col3.metric(
            "Гапаксы (только Заболоцкий)",
            f"{hapax_metadata.get('hapax_count', 0):,}".replace(',', ' ')
        )
        col4.metric(
            "% от словаря",
            f"{100 * hapax_metadata.get('hapax_count', 0) / max(1, hapax_metadata.get('zabolotsky_unique_lemmas', 1)):.1f}%"
        )

        st.divider()

        # Получаем однократные гапаксы (встречаются в корпусе ровно один раз)
        hapax_once = get_hapax_legomena(full_corpus, hapax_set)

        st.markdown(f"### Однократные единицы: {len(hapax_once):,}")

        if len(hapax_once) == 0:
            st.info("Нет однократных единиц!")
        else:
            
            hapax_list = sorted(hapax_once.keys())
            in_navec, not_in_navec = check_hapax_in_navec(hapax_list)

            tab_all_hapax, tab_unknown_vectors = st.tabs([
                f"📚 Все однократные ({len(hapax_list)})",
                f"⚠️ Отсутствуют в navec ({len(not_in_navec)})"
            ])

            # Таблица 1: Все однократные неологизмы
            with tab_all_hapax:
                hapax_df = pd.DataFrame({"Слово": hapax_list})
                hapax_df.index = range(1, len(hapax_df) + 1)

                st.dataframe(hapax_df, width='stretch', height=500)

                st.divider()

                # Скачивание файла
                csv = hapax_df.to_csv(index_label='№')
                st.download_button(
                    label="📥 Скачать список CSV",
                    data=csv,
                    file_name="mayak_hapax_legomena.csv",
                    mime="text/csv"
                )

            # Таблица 2: Слова отсутствующие в navec
            with tab_unknown_vectors:
                st.markdown(
                    "Эти слова встречаются у Заболоцкого только один раз"
                    "и отсутствуют в векторной модели Navec (вероятно, опечатки или уникальные авторские неологизмы)."
                )

                if len(not_in_navec) == 0:
                    st.success("✅ Все однократные неологизмы представлены в модели navec")
                else:
                    unknown_df = pd.DataFrame({"Слово": not_in_navec})
                    unknown_df.index = range(1, len(unknown_df) + 1)

                    st.dataframe(unknown_df, width='stretch', height=500)

                    st.divider()

                    # Скачивание файла
                    csv_unknown = unknown_df.to_csv(index_label='№')
                    st.download_button(
                        label="📥 Скачать список CSV",
                        data=csv_unknown,
                        file_name="mayak_unknown_vectors.csv",
                        mime="text/csv"
                    )

        st.divider()

        with st.expander(f"📚 Все гапаксы ({len(hapax_set):,})"):
            st.markdown("*(слова, которые встречаются только у Заболоцкого, независимо от частоты)*")

            freq_counter = compute_frequency_dict(full_corpus, exclude_stopwords=False)

            all_hapax_list = [
                {"Слово": lemma, "Частота": freq_counter.get(lemma, 0)}
                for lemma in sorted(hapax_set, key=lambda x: freq_counter.get(x, 0), reverse=True)
            ]

            all_hapax_df = pd.DataFrame(all_hapax_list)
            all_hapax_df.index = range(1, len(all_hapax_df) + 1)

            st.dataframe(all_hapax_df, width='stretch', height=600)

            # Статистика по частотности
            st.markdown("#### Распределение по частотности")
            freq_counts = Counter(h['Частота'] for h in all_hapax_list)
            freq_dist_df = pd.DataFrame([
                {"Частота": freq, "Количество слов": count}
                for freq, count in sorted(freq_counts.items())
            ])

            col_table, col_chart = st.columns([0.5, 0.5])
            with col_table:
                st.dataframe(freq_dist_df, width='stretch')
            with col_chart:
                st.bar_chart(freq_dist_df.set_index('Частота')['Количество слов'])

