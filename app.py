import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import json
import warnings
import pickle
from io import BytesIO
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from PIL import Image

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    GRU, SimpleRNN, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

st.set_page_config(
    page_title="Stress Detection NLP Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main { padding-top: 2rem; }
h1 { color: #1f77b4; }
.stButton>button { color: white; background-color: #1f77b4; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONFIG
# ============================================
DATA_PATH = "Stress.csv"
MODELS_DIR = "streamlit_models"
os.makedirs(MODELS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(MODELS_DIR, "results.json")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "keras_tokenizer.pkl")
VOCAB_SIZE = 15000
MAX_LENGTH = 150
EMBEDDING_DIM = 128

# ============================================
# DATA PIPELINE
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class PublicImageParser(HTMLParser):
    """Find a page's social-preview image, then fall back to its first image."""

    def __init__(self):
        super().__init__()
        self.preview_images = []
        self.images = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "meta":
            name = (attributes.get("property") or attributes.get("name") or "").lower()
            if name in {"og:image", "twitter:image", "twitter:image:src"}:
                if attributes.get("content"):
                    self.preview_images.append(attributes["content"])
        elif tag == "img" and attributes.get("src"):
            self.images.append(attributes["src"])


def download_public_url(url):
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Enter a valid public http(s) URL.")

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=10) as response:
        payload = response.read(5 * 1024 * 1024 + 1)
        content_type = response.headers.get_content_type()
        resolved_url = response.geturl()

    if len(payload) > 5 * 1024 * 1024:
        raise ValueError("The image must be 5 MB or smaller.")
    return payload, content_type, resolved_url


@st.cache_data(show_spinner=False)
def image_bytes_to_mask(image_bytes):
    """Convert an image into a WordCloud mask."""
    image = Image.open(BytesIO(image_bytes)).convert("L")
    image.thumbnail((800, 800))
    mask = np.array(image)

    # In WordCloud masks, dark pixels are drawable and white pixels are excluded.
    return np.where(mask < 250, 0, 255).astype("uint8")


@st.cache_data(show_spinner=False)
def image_url_to_mask(image_url):
    """Create a WordCloud mask from a direct image or public webpage link."""
    image_bytes, content_type, resolved_url = download_public_url(image_url)

    if not content_type.startswith("image/"):
        parser = PublicImageParser()
        parser.feed(image_bytes.decode("utf-8", errors="ignore"))
        candidate_images = parser.preview_images + parser.images
        if not candidate_images:
            raise ValueError("No public preview image was found at this link.")
        image_bytes, _, _ = download_public_url(
            urljoin(resolved_url, candidate_images[0])
        )

    return image_bytes_to_mask(image_bytes)


@st.cache_data
def preprocess_data(df):
    data = df[['text', 'label']].dropna().copy()
    data['text'] = data['text'].apply(clean_text)
    data['label'] = data['label'].astype(int)
    data = data.reset_index(drop=True)
    return data


@st.cache_data
def get_split(data):
    X = data['text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


df = load_data()
data = preprocess_data(df)
X_train, X_test, y_train, y_test = get_split(data)


@st.cache_resource
def get_tokenizer():
    tok = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tok.fit_on_texts(X_train)
    return tok


@st.cache_resource
def get_tfidf():
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    vec.fit(X_train)
    return vec


@st.cache_data
def get_sequences(_tokenizer, texts):
    seqs = _tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LENGTH, padding='post', truncating='post')


tokenizer = get_tokenizer()
tfidf = get_tfidf()

X_train_seq = get_sequences(tokenizer, X_train)
X_test_seq = get_sequences(tokenizer, X_test)
X_train_tfidf = tfidf.transform(X_train).toarray().astype('float32')
X_test_tfidf = tfidf.transform(X_test).toarray().astype('float32')

y_train_np = y_train.values.astype('float32')
y_test_np = y_test.values.astype('float32')

# ============================================
# MODEL PERSISTENCE HELPERS
# ============================================
def model_path(name):
    return os.path.join(MODELS_DIR, f"{name}.keras")


def history_path(name):
    return os.path.join(MODELS_DIR, f"{name}_history.json")


def is_saved(name):
    return os.path.exists(model_path(name))


def save_results(results):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f)


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_history(name, hist):
    payload = {
        'accuracy': [float(x) for x in hist.history.get('accuracy', [])],
        'val_accuracy': [float(x) for x in hist.history.get('val_accuracy', [])],
        'loss': [float(x) for x in hist.history.get('loss', [])],
        'val_loss': [float(x) for x in hist.history.get('val_loss', [])],
    }
    with open(history_path(name), 'w') as f:
        json.dump(payload, f)


def load_history(name):
    p = history_path(name)
    if os.path.exists(p):
        with open(p, 'r') as f:
            return json.load(f)
    return None


# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("🧠 Stress Detection NLP")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "📊 About Dataset",
        "🧹 Preprocessing",
        "🤖 Models & Training",
        "📈 Comparison Results",
        "🏆 Best Model",
        "🧪 Test Model",
        "📥 Download Model",
    ],
)

# ============================================
# PAGE: ABOUT DATASET
# ============================================
if page == "📊 About Dataset":
    st.title("📊 About the Dataset")
    st.markdown(f"**Source file:** `{DATA_PATH}`")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Duplicated Rows", df.duplicated().sum())

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing Values per Column")
        fig, ax = plt.subplots()
        df.isnull().sum().plot(kind='bar', ax=ax, color='coral')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Label Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='label', hue='label', legend=False, palette='Set2', ax=ax)
        ax.set_xlabel("Label (0 = No Stress, 1 = Stress)")
        ax.set_title("Stress Label Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Types")
        st.write(df.dtypes)
    with col2:
        st.subheader("Numerical Summary")
        st.write(df.describe())

    if 'subreddit' in df.columns:
        st.subheader("Most Common Subreddits")
        subreddit_counts = df['subreddit'].dropna().value_counts()
        top_subreddits = subreddit_counts.head(15).sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=top_subreddits.values,
            y=top_subreddits.index,
            hue=top_subreddits.index,
            palette="crest",
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Posts")
        ax.set_ylabel("Subreddit")
        ax.set_title("Top 15 Subreddits by Number of Posts")
        st.pyplot(fig)
        plt.close()

        with st.expander("View all subreddit counts"):
            st.dataframe(
                subreddit_counts.rename_axis("Subreddit").reset_index(name="Posts"),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Word Cloud - Entire Dataset")
    st.caption("Upload an image to shape the word cloud, or use a public image/webpage link instead.")
    mask_image = st.file_uploader(
        "Upload an image to use as the word-cloud mask (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        help="Use a dark subject on a white background. The uploaded image takes priority over the URL below.",
    )
    if mask_image is not None:
        st.image(mask_image, caption="Selected word-cloud mask", width=240)
    image_url = st.text_input(
        "Public image or webpage link for word-cloud shape (optional fallback)",
        placeholder="https://example.com/image.png or https://example.com/article",
        help="For webpages, the app uses the page preview image or first image. Use a dark subject on a white background.",
    )
    if st.button("Generate Word Cloud", type="primary"):
        with st.spinner("Generating word cloud..."):
            from wordcloud import WordCloud as WC
            from nltk.corpus import stopwords

            try:
                if mask_image is not None:
                    mask = image_bytes_to_mask(mask_image.getvalue())
                else:
                    mask = image_url_to_mask(image_url) if image_url.strip() else None
                all_words = " ".join(data['text'].tolist())
                stop_words = set(stopwords.words('english'))
                wordcloud = WC(
                    width=800,
                    height=400,
                    background_color="white",
                    stopwords=stop_words,
                    collocations=False,
                    mask=mask,
                ).generate(all_words)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                ax.set_title("Word Cloud for Entire Dataset")
                st.pyplot(fig)
                plt.close()
            except Exception as error:
                st.error(f"Unable to generate the word cloud: {error}")

# ============================================
# PAGE: PREPROCESSING
# ============================================
elif page == "🧹 Preprocessing":
    st.title("🧹 Data Preprocessing")

    st.subheader("1. Text Cleaning")
    sample = st.text_area("Sample text from dataset:", value=data['text'].iloc[0], height=120)
    if st.button("Clean Text"):
        cleaned = clean_text(sample)
        st.code(cleaned)

    st.subheader("2. Tokenization & Stopword Removal")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show Tokenization"):
            import nltk
            nltk.download('punkt', quiet=True)
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(sample)
            st.write(tokens)
    with col2:
        if st.button("Show Stopword Removal"):
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in sample.split() if w.lower() not in stop_words]
            st.write(tokens)

    st.subheader("3. Train / Test Split")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Train Samples", len(X_train))
    with col2:
        st.metric("Test Samples", len(X_test))
    with col3:
        st.metric("Vocabulary Size", len(tokenizer.word_index))

    if st.button("Show Padding Example"):
        ex = X_test.iloc[0]
        seq = tokenizer.texts_to_sequences([ex])[0]
        pad = pad_sequences([seq], maxlen=MAX_LENGTH, padding='post')
        st.write("**Original:**", ex)
        st.write("**Tokens (first 20):**", seq[:20])
        st.write("**Padded shape:**", pad.shape)
        st.write("**Padded (first 20):**", pad[0][:20])

# ============================================
# PAGE: MODELS & TRAINING
# ============================================
elif page == "🤖 Models & Training":
    st.title("🤖 Models & Training")
    st.markdown("""
    This section trains 5 deep learning models on the stress detection dataset:
    1. **ANN** — TF-IDF features → Dense layers
    2. **Simple RNN** — Embedding → SimpleRNN
    3. **LSTM** — Embedding → LSTM
    4. **GRU** — Embedding → GRU
    5. **Bidirectional LSTM** — Embedding → BiLSTM → Dense layers
    """)

    status_rows = []
    for m in ['ANN', 'RNN', 'LSTM', 'GRU', 'BiLSTM']:
        status_rows.append({
            'Model': m,
            'Status': '✅ Saved' if is_saved(m) else '❌ Not Trained'
        })
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    if st.button("🚀 Train Missing Models", type="primary"):
        progress = st.progress(0)
        status = st.empty()
        results = load_results()
        histories = {}

        progress_bar = [0]

        def advance(name, pct):
            progress.progress(pct)
            status.text(f"Training {name}...")

        # ---- ANN ----
        if not is_saved('ANN'):
            advance('ANN (TF-IDF)', 10)
            ann = Sequential([
                Dense(256, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
                Dropout(0.4),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid'),
            ])
            ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            es = EarlyStopping(patience=2, restore_best_weights=True, verbose=0)
            hist = ann.fit(
                X_train_tfidf, y_train_np,
                validation_data=(X_test_tfidf, y_test_np),
                epochs=10, batch_size=32, callbacks=[es], verbose=0
            )
            ann.save(model_path('ANN'))
            save_history('ANN', hist)
            p = ann.predict(X_test_tfidf, verbose=0)
            results['ANN'] = float(accuracy_score(y_test_np, (p > 0.5).astype(int)))
            histories['ANN'] = hist
        else:
            if 'ANN' not in results:
                p = load_model(model_path('ANN')).predict(X_test_tfidf, verbose=0)
                results['ANN'] = float(accuracy_score(y_test_np, (p > 0.5).astype(int)))

        advance('Simple RNN', 30)

        # ---- RNN ----
        if not is_saved('RNN'):
            rnn = Sequential([
                Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
                SimpleRNN(64, dropout=0.2),
                Dense(1, activation='sigmoid'),
            ])
            rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            es = EarlyStopping(patience=2, restore_best_weights=True, verbose=0)
            hist = rnn.fit(
                X_train_seq, y_train_np,
                validation_data=(X_test_seq, y_test_np),
                epochs=8, batch_size=32, callbacks=[es], verbose=0
            )
            rnn.save(model_path('RNN'))
            save_history('RNN', hist)
            p = rnn.predict(X_test_seq, verbose=0)
            results['RNN'] = float(accuracy_score(y_test_np, (p > 0.5).astype(int)))
            histories['RNN'] = hist
        else:
            if 'RNN' not in results:
                p = load_model(model_path('RNN')).predict(X_test_seq, verbose=0)
                results['RNN'] = float(accuracy_score(y_test_np, (p > 0.5).astype(int)))

        advance('LSTM', 50)

        # ---- LSTM ----
        if not is_saved('LSTM'):
            lstm = Sequential([
                Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
                LSTM(64, dropout=0.25),
                Dense(1, activation='sigmoid'),
            ])
            lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            es = EarlyStopping(patience=2, restore_best_weights=True, verbose=0)
            hist = lstm.fit(
                X_train_seq, y_train_np,
                validation_data=(X_test_seq, y_test_np),
                epochs=8, batch_size=32, callbacks=[es], verbose=0
            )
            lstm.save(model_path('LSTM'))
            save_history('LSTM', hist)
            p = lstm.predict(X_test_seq, verbose=0)
            results['LSTM'] = float(accuracy_score(y_test_np, (p > 0.5).astype(int)))
            histories['LSTM'] = hist
        else:
            if 'LSTM' not in results:
                p = load_model(model_path('LSTM')).predict(X_test_seq, verbose=0)
                results['LSTM'] = float(accuracy_score(y_test_np, (p > 0.5).astype(int)))

        advance('GRU', 70)

        # ---- GRU ----
        if not is_saved('GRU'):
            gru = Sequential([
                Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
                GRU(64, dropout=0.25),
                Dense(1, activation='sigmoid'),
            ])
            gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            es = EarlyStopping(patience=2, restore_best_weights=True, verbose=0)
            hist = gru.fit(
                X_train_seq, y_train_np,
                validation_data=(X_test_seq, y_test_np),
                epochs=8, batch_size=32, callbacks=[es], verbose=0
            )
            gru.save(model_path('GRU'))
            save_history('GRU', hist)
            p = gru.predict(X_test_seq, verbose=0)
            results['GRU'] = float(accuracy_score(y_test_np, (p > 0.5).astype(int)))
            histories['GRU'] = hist
        else:
            if 'GRU' not in results:
                p = load_model(model_path('GRU')).predict(X_test_seq, verbose=0)
                results['GRU'] = float(accuracy_score(y_test_np, (p > 0.5).astype(int)))

        advance('Bidirectional LSTM', 85)

        # ---- BiLSTM ----
        if not is_saved('BiLSTM'):
            bilstm = Sequential([
                Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH, mask_zero=True),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dropout(0.3),
                Bidirectional(LSTM(32)),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid'),
            ])
            bilstm.compile(
                optimizer=Adam(learning_rate=0.0005),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            es = EarlyStopping(patience=3, restore_best_weights=True, verbose=0)
            hist = bilstm.fit(
                X_train_seq, y_train_np,
                validation_data=(X_test_seq, y_test_np),
                epochs=10, batch_size=32, callbacks=[es], verbose=0
            )
            bilstm.save(model_path('BiLSTM'))
            save_history('BiLSTM', hist)
            p = bilstm.predict(X_test_seq, verbose=0)
            results['BiLSTM'] = float(accuracy_score(y_test_np, (p > 0.5).astype(int)))
            histories['BiLSTM'] = hist
        else:
            if 'BiLSTM' not in results:
                p = load_model(model_path('BiLSTM')).predict(X_test_seq, verbose=0)
                results['BiLSTM'] = float(accuracy_score(y_test_np, (p > 0.5).astype(int)))

        advance('Finalizing', 100)
        status.text("✅ All models trained and saved!")
        save_results(results)
        st.session_state['results'] = results
        st.session_state['histories'] = histories
        st.session_state['models_trained'] = True

    if st.session_state.get('models_trained', False):
        st.success("Models are trained! Check 'Comparison Results' for metrics.")
        st.json(st.session_state['results'])

# ============================================
# PAGE: COMPARISON RESULTS
# ============================================
elif page == "📈 Comparison Results":
    st.title("📈 Comparison Results")

    results = load_results()
    if not results:
        st.warning("No results found. Go to 'Models & Training' to train models first.")
    else:
        df_results = pd.DataFrame(
            list(results.items()), columns=['Model', 'Test Accuracy']
        ).sort_values('Test Accuracy', ascending=False).reset_index(drop=True)
        df_results['Test Accuracy'] = df_results['Test Accuracy'].apply(lambda x: f"{x:.2%}")

        st.subheader("🏆 Final Accuracy Ranking")
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=df_results,
            x='Model',
            y=df_results['Test Accuracy'].str.rstrip('%').astype(float),
            palette='viridis',
            ax=ax,
        )
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("Model Comparison - Test Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Training / Validation Curves")
    selected_model = st.selectbox(
        "Select a model to view its curves:",
        ['ANN', 'RNN', 'LSTM', 'GRU', 'BiLSTM']
    )
    hist = load_history(selected_model)
    if hist:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(hist['accuracy'], label='Training')
            ax.plot(hist['val_accuracy'], label='Validation')
            ax.set_title(f'{selected_model} Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            plt.close()
        with col2:
            fig, ax = plt.subplots()
            ax.plot(hist['loss'], label='Training')
            ax.plot(hist['val_loss'], label='Validation')
            ax.set_title(f'{selected_model} Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            plt.close()
    else:
        st.write("No history found for this model.")

# ============================================
# PAGE: BEST MODEL
# ============================================
elif page == "🏆 Best Model":
    st.title("🏆 Best Model")

    results = load_results()
    if not results:
        st.warning("Train models first to see the best model.")
    else:
        best_model_name = max(results, key=results.get)
        best_accuracy = results[best_model_name]

        st.subheader("Recommended Model")
        st.metric("Best Model", best_model_name)
        st.metric("Test Accuracy", f"{best_accuracy:.2%}")

        st.subheader("Why Use This Model?")
        rationale = {
            'ANN': "Lightweight and fast to train. Good for quick baseline predictions.",
            'RNN': "Simple recurrent model. May capture short-term dependencies.",
            'LSTM': "Handles long-range dependencies better than basic RNNs.",
            'GRU': "Fewer parameters than LSTM but similar performance in many cases.",
            'BiLSTM': (
                "Bidirectional LSTM captures context from both directions, "
                "typically highest accuracy among classical architectures."
            ),
        }
        st.markdown(rationale.get(best_model_name, "Performs best on held-out test data."))

        st.subheader("Classification Report on Test Set")
        if is_saved(best_model_name):
            model = load_model(model_path(best_model_name))
            if best_model_name == 'ANN':
                preds = model.predict(X_test_tfidf, verbose=0)
            else:
                preds = model.predict(X_test_seq, verbose=0)
            y_pred = (preds > 0.5).astype(int).flatten()
            report = classification_report(y_test_np, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test_np, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title(f'{best_model_name} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)
            plt.close()

# ============================================
# PAGE: TEST MODEL
# ============================================
elif page == "🧪 Test Model":
    st.title("🧪 Test Model")

    results = load_results()
    if not results:
        st.warning("Train models first.")
    else:
        model_option = st.selectbox(
            "Choose a model to test:",
            list(results.keys())
        )

        user_input = st.text_area(
            "Enter text to classify:",
            height=120,
            placeholder="Type or paste text here..."
        )

        if st.button("Predict", type="primary"):
            if not user_input.strip():
                st.warning("Please enter some text.")
            else:
                cleaned = clean_text(user_input)
                seq = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')

                if model_option == 'ANN':
                    tfidf_input = tfidf.transform([cleaned]).toarray().astype('float32')
                    prob = float(load_model(model_path('ANN')).predict(tfidf_input, verbose=0)[0][0])
                else:
                    prob = float(load_model(model_path(model_option)).predict(padded, verbose=0)[0][0])

                label = "Stress" if prob >= 0.5 else "No Stress"
                confidence = prob if prob >= 0.5 else 1 - prob

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", label)
                with col2:
                    st.metric("Probability (Stress)", f"{prob:.4f}")
                with col3:
                    st.metric("Confidence", f"{confidence:.2%}")

                if prob >= 0.5:
                    st.error("⚠️ This text indicates potential stress.")
                else:
                    st.success("✅ This text appears non-stressful.")

# ============================================
# PAGE: DOWNLOAD MODEL
# ============================================
elif page == "📥 Download Model":
    st.title("📥 Download Model")

    results = load_results()
    if not results:
        st.warning("Train models first.")
    else:
        best_model_name = max(results, key=results.get)
        st.info(f"Best model: **{best_model_name}** (used for download by default).")

        download_name = st.selectbox(
            "Or select a specific model to download:",
            list(results.keys()),
            index=list(results.keys()).index(best_model_name)
        )

        if is_saved(download_name):
            model_file = model_path(download_name)
            with open(model_file, 'rb') as f:
                model_bytes = f.read()

            st.download_button(
                label=f"⬇️ Download {download_name} (.keras)",
                data=model_bytes,
                file_name=f"{download_name.lower()}_model.keras",
                mime="application/octet-stream",
            )

            st.subheader("Tokenizer")
            try:
                with open(TOKENIZER_PATH, 'rb') as f:
                    tok_bytes = f.read()
                st.download_button(
                    label="⬇️ Download Tokenizer (.pkl)",
                    data=tok_bytes,
                    file_name="keras_tokenizer.pkl",
                    mime="application/octet-stream",
                )
            except FileNotFoundError:
                with open(TOKENIZER_PATH, 'wb') as f:
                    pickle.dump(tokenizer, f)
                with open(TOKENIZER_PATH, 'rb') as f:
                    tok_bytes = f.read()
                st.download_button(
                    label="⬇️ Download Tokenizer (.pkl)",
                    data=tok_bytes,
                    file_name="keras_tokenizer.pkl",
                    mime="application/octet-stream",
                )

            if download_name == 'ANN':
                tfidf_file = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
                if not os.path.exists(tfidf_file):
                    with open(tfidf_file, 'wb') as f:
                        pickle.dump(tfidf, f)
                with open(tfidf_file, 'rb') as f:
                    vec_bytes = f.read()
                st.download_button(
                    label="⬇️ Download TF-IDF Vectorizer (.pkl)",
                    data=vec_bytes,
                    file_name="tfidf_vectorizer.pkl",
                    mime="application/octet-stream",
                )
        else:
            st.warning("This model hasn't been saved yet. Train it first.")
