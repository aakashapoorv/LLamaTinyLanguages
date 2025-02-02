import streamlit as st
import os
import base64
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize Streamlit
st.set_page_config(page_title="LlamaSmallLanguages", layout="wide")

# Logo file (if it exists)
LOGO_PATH = "logo.png"

# Convert image to base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Check if logo exists
base64_logo = None
if os.path.exists(LOGO_PATH):
    base64_logo = get_base64_of_image(LOGO_PATH)

# Model paths
MODEL_PATHS = {
    "English": "OpenGenerativeAI/Llama-3.2-3B-Sami_Kven_Faroese_Kalaallisut-Instruct-16bit",
    "Kalaallisut": "OpenGenerativeAI/Llama-3.2-3B-Sami_Kven_Faroese_Kalaallisut-Instruct-16bit",
    "Kveeni": "OpenGenerativeAI/Llama-3.2-3B-Sami_Kven_Faroese_Kalaallisut-Instruct-16bit",
    "Sámegiella": "OpenGenerativeAI/Llama-3.2-3B-Sami_Kven_Faroese_Kalaallisut-Instruct-16bit",
    "Føroyskt": "OpenGenerativeAI/Llama-3.2-3B-Sami_Kven_Faroese_Kalaallisut-Instruct-16bit"
}

# Language selection and descriptions
LANGUAGES = {
    "English": "🌍 Multilingual Chatbot",
    "Kalaallisut": "🌍 Oqaatsinut ikittunut ikiuisartoq Chatbot",
    "Kveeni": "🌍 Monikielinen Chatbot",
    "Sámegiella": "🌍 Máŋggagealá Chatbot",
    "Føroyskt": "🌍 Fleirmáls Chatbot"
}

DESCRIPTIONS = {
    "English": "<p>These Nordic indigenous (minority) languages carry cultural identity and heritage. There is very little training data available on them, so we are helping by releasing more datasets in these languages and creating a Llama model for communities that may not have access to high-speed internet or high-performance GPUs. Our model can run on low-cost consumer hardware.</p>",
    "Kalaallisut": "<p>Oqaatsit ikittuusut sumiiffinni kulturiinik ersersitsipput. Oqaatsinut taakkununnga sungiusaatigalugu paasissutissanik annikitsuinnaasunik peqarpoq, taamaattumik uagut oqaatsinik taakkununnga paasissutissanik annertusisitsilluta ikiueqqissaagut.</p>",
    "Kveeni": "<p>Nämä Pohjoismaiden vähemmistökielet kantavat kulttuurista identiteettiä ja perintöä. Niille on saatavilla vain vähän koulutusdataa, joten autamme julkaisemalla lisää datasettejä ja luomalla Llama-mallin yhteisöille, joilla ei ole pääsyä nopeaan internettiin tai tehokkaisiin GPU:hin.</p>",
    "Sámegiella": "<p>Dát máŋggagealá gielat čájehit kultuvrra identitehta ja árbbedieđuid. Oassálastaba unnán oahppanmateriála, nu ahte mii veahkehit lassin materjálain ja ráhkadit Llama-modela dálkkádatgieldiin geat eai sáhte geavahit fástain internettain dahje divrra GPU:in.</p>",
    "Føroyskt": "<p>Hesi mál bera mentanarliga samleika og arv. Nógv lítið av dátum finnast til hesi mál, og tí hjálpa vit við at útgeva fleiri dátusett og at menna ein Llama-máldátumodell til samfeløg, sum ikki hava atgongd til skjótt internet ella sterkar GPU-ir.</p>"
}

PLACEHOLDERS = {
    "English": "Type your message here...",
    "Kalaallisut": "Allagit nalunaarutit uunga...",
    "Kveeni": "Kirjoita viestisi tähän...",
    "Sámegiella": "Čállá sániid dáppe...",
    "Føroyskt": "Skriva tín boðskap her..."
}

# Load models and tokenizers
models = {}
tokenizers = {}

for lang, model_path in MODEL_PATHS.items():
    try:
        models[lang] = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        tokenizers[lang] = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model for {lang}: {e}")
        models[lang] = None
        tokenizers[lang] = None

# Ensure session state variables exist
if "lang_state" not in st.session_state:
    st.session_state.lang_state = "English"
if "histories_state" not in st.session_state:
    st.session_state.histories_state = {lang: [] for lang in LANGUAGES}
if "messages" not in st.session_state:
    st.session_state.messages = st.session_state.histories_state.get(st.session_state.lang_state, [])

# Sidebar
with st.sidebar:
    if base64_logo:
        st.markdown(f"""
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{base64_logo}" width="250">
            </div>
        """, unsafe_allow_html=True)

    st.title(LANGUAGES[st.session_state.lang_state])  # Dynamic title based on language
    st.markdown(DESCRIPTIONS[st.session_state.lang_state], unsafe_allow_html=True)

    st.subheader("Settings")

    temperature = st.slider("Temperature", min_value=0.01, max_value=2.0, value=0.7, step=0.01)
    top_p = st.slider("Top P", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider("Max Length", min_value=64, max_value=2048, value=512, step=8)

    # Reset chat history
    def clear_chat_history():
        st.session_state.messages = []
        st.session_state.histories_state[st.session_state.lang_state] = []
        st.rerun()

    st.button("🗑️ Reset Chat", on_click=clear_chat_history)

# Language selection buttons
cols = st.columns(len(LANGUAGES))
for idx, lang in enumerate(LANGUAGES.keys()):
    if cols[idx].button(lang):
        st.session_state.lang_state = lang
        st.session_state.messages = st.session_state.histories_state.get(lang, [])
        st.rerun()

# Chat window
st.markdown("## 💬 Chat")

chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.write("...")

    for message in st.session_state.messages:
        role = "🧑‍💻 User" if message["role"] == "user" else "🤖 AI"
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Chat input
user_input = st.chat_input(PLACEHOLDERS[st.session_state.lang_state])

# Generate responses
def chat_with_local_llm(user_input, histories, language):
    if not user_input.strip():
        return histories[language], histories

    model = models.get(language)
    tokenizer = tokenizers.get(language)

    if model is None or tokenizer is None:
        assistant_reply = f"🤖 No model available for {language}. Please check the configuration."
    else:
        input_ids = tokenizer.encode(user_input, return_tensors="pt").to("cuda")

        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

        assistant_reply = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    histories[language].append({"role": "user", "content": user_input})
    histories[language].append({"role": "assistant", "content": assistant_reply})

    return histories[language], histories

# Process user input
if user_input is not None and user_input.strip():
    st.session_state.messages, st.session_state.histories_state = chat_with_local_llm(
        user_input, st.session_state.histories_state, st.session_state.lang_state
    )
    st.rerun()
