import json
import re

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FAQChatbot:

    def __init__(self, faq_file):

        # ====================================================
        # FAQ FILE
        # ====================================================

        self.faq_file = Path(faq_file)


        if not self.faq_file.exists():

            self.faq_file = (
                Path(__file__).resolve().parent.parent
                / faq_file
            )


        if not self.faq_file.exists():

            raise FileNotFoundError(
                f"FAQ file not found: {self.faq_file}"
            )


        # ====================================================
        # LOAD JSON
        # ====================================================

        with open(
            self.faq_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)


        # ====================================================
        # SUPPORT JSON FORMATS
        # ====================================================

        if isinstance(data, dict):

            if "faqs" in data:

                data = data["faqs"]

            elif "questions" in data:

                data = data["questions"]

            else:

                data = [data]


        if not isinstance(data, list):

            raise ValueError(
                "faq_data.json must contain a list."
            )


        self.questions = []

        self.answers = []


        # ====================================================
        # READ FAQS
        # ====================================================

        for item in data:

            if not isinstance(item, dict):

                continue


            question = (
                item.get("question")
                or item.get("Question")
                or item.get("q")
                or ""
            )


            answer = (
                item.get("answer")
                or item.get("Answer")
                or item.get("a")
                or ""
            )


            question = str(
                question
            ).strip()


            answer = str(
                answer
            ).strip()


            if question and answer:

                self.questions.append(
                    question
                )

                self.answers.append(
                    answer
                )


        if not self.questions:

            raise ValueError(
                "No valid FAQs found in faq_data.json"
            )


        # ====================================================
        # TF-IDF
        # ====================================================

        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocess_text,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            sublinear_tf=True
        )


        self.faq_vectors = (
            self.vectorizer.fit_transform(
                self.questions
            )
        )


        print(
            f"✅ Loaded {len(self.questions)} FAQs."
        )


    # ========================================================
    # PREPROCESS TEXT
    # ========================================================

    def preprocess_text(self, text):

        text = str(text).lower()


        # Remove URLs

        text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            text
        )


        # Remove special characters

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )


        # Remove extra spaces

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()


        # Simple tokenization

        words = text.split()


        # Stopwords

        stop_words = {

            "a",
            "an",
            "the",
            "is",
            "are",
            "am",
            "was",
            "were",
            "be",
            "been",
            "being",

            "to",
            "of",
            "in",
            "on",
            "for",
            "with",

            "and",
            "or",
            "but",

            "this",
            "that",
            "these",
            "those",

            "it",
            "its",

            "as",
            "at",
            "by",
            "from",

            "can",
            "could",
            "would",
            "should",

            "do",
            "does",
            "did",

            "what",
            "who",
            "why",
            "how"
        }


        words = [
            word
            for word in words
            if word not in stop_words
        ]


        return words


    # ========================================================
    # GET RESPONSE
    # ========================================================

    def get_response(
        self,
        user_question,
        threshold=0.10
    ):

        user_question = str(
            user_question
        ).strip()


        if not user_question:

            return {

                "answer":
                    "Please enter a question.",

                "similarity":
                    0.0,

                "question":
                    ""

            }


        # ====================================================
        # USER VECTOR
        # ====================================================

        user_vector = (
            self.vectorizer.transform(
                [user_question]
            )
        )


        # ====================================================
        # COSINE SIMILARITY
        # ====================================================

        similarities = cosine_similarity(
            user_vector,
            self.faq_vectors
        )[0]


        # Best match

        best_index = int(
            similarities.argmax()
        )


        best_score = float(
            similarities[best_index]
        )


        matched_question = (
            self.questions[best_index]
        )


        matched_answer = (
            self.answers[best_index]
        )


        # ====================================================
        # LOW CONFIDENCE
        # ====================================================

        if best_score < threshold:

            return {

                "answer":
                    "Sorry, I couldn't find a relevant answer in the FAQ knowledge base.",

                "similarity":
                    round(best_score, 4),

                "question":
                    matched_question

            }


        # ====================================================
        # SUCCESS
        # ====================================================

        return {

            "answer":
                matched_answer,

            "similarity":
                round(best_score, 4),

            "question":
                matched_question

        }