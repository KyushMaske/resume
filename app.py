import streamlit as st
import pickle
import re
import nltk


nltk.download("punkt")
nltk.download("stopwords")


def clean_resume(resume_text):
    clean_text = re.sub("http\S+\s*", " ", resume_text)
    clean_text = re.sub("RT|cc", " ", clean_text)
    clean_text = re.sub("#\S+", "", clean_text)
    clean_text = re.sub("@\S+", "  ", clean_text)
    clean_text = re.sub(
        "[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", clean_text
    )
    clean_text = re.sub(r"[^\x00-\x7f]", r" ", clean_text)
    clean_text = re.sub("\s+", " ", clean_text)
    return clean_text


# Load models
clf = pickle.load(open("model.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
onehot_encoder = pickle.load(open("onehot_encoder.pkl", "rb"))


# Web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode("utf-8")
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode("latin-1")

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf_vectorizer.transform([cleaned_resume])
        prediction = clf.predict(input_features)
        prediction_id = onehot_encoder.inverse_transform(prediction)[0][0]

        st.write("Predicted Category:", prediction_id)


if __name__ == "__main__":
    main()
