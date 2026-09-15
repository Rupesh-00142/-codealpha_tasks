from flask import Flask, request, jsonify
from flask_cors import CORS

from chatbot.faq_engine import FAQChatbot


app = Flask(__name__)

# Allow frontend on port 5500
CORS(
    app,
    resources={
        r"/ask": {
            "origins": [
                "http://127.0.0.1:5500",
                "http://localhost:5500"
            ]
        }
    }
)


# ============================================================
# LOAD CHATBOT
# ============================================================

try:

    chatbot = FAQChatbot("faq_data.json")

    print("✅ FAQ chatbot loaded successfully.")

except Exception as e:

    print("❌ Chatbot loading error:")
    print(e)

    chatbot = None


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return """
    <h2>FAQ AI Assistant Backend</h2>
    <p>Backend is running successfully.</p>
    <p>API: POST /ask</p>
    """


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "online",
        "chatbot": chatbot is not None
    })


# ============================================================
# ASK
# ============================================================

@app.route("/ask", methods=["POST"])
def ask():

    try:

        # Check chatbot

        if chatbot is None:

            return jsonify({
                "error": "FAQ chatbot failed to initialize."
            }), 500


        # Read JSON

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({
                "error": "Invalid JSON request."
            }), 400


        # Get question

        user_question = str(
            data.get("question", "")
        ).strip()


        if not user_question:

            return jsonify({
                "error": "Please enter a question."
            }), 400


        print()
        print("=" * 60)
        print("USER:")
        print(user_question)


        # ====================================================
        # GET CHATBOT RESPONSE
        # ====================================================

        result = chatbot.get_response(
            user_question
        )


        print("RAW RESULT:")
        print(result)


        # ====================================================
        # HANDLE DIFFERENT RESULT FORMATS
        # ====================================================

        if isinstance(result, dict):

            answer = (
                result.get("answer")
                or result.get("response")
                or result.get("text")
                or "Sorry, I could not find an answer."
            )


            similarity = result.get(
                "similarity",
                result.get("score", 0)
            )


            matched_question = (
                result.get("question")
                or result.get("matched_question")
                or ""
            )


        else:

            answer = str(result)

            similarity = 0

            matched_question = ""


        # Make similarity safe

        try:

            similarity = float(
                similarity
            )

        except:

            similarity = 0.0


        # ====================================================
        # RESPONSE TO FRONTEND
        # ====================================================

        response_data = {

            "answer": answer,

            "similarity":
                round(similarity, 4),

            "question":
                matched_question

        }


        print("ANSWER:")
        print(answer)

        print("SIMILARITY:")
        print(similarity)

        print("=" * 60)


        return jsonify(
            response_data
        )


    except Exception as e:

        print()
        print("=" * 60)
        print("❌ ERROR IN /ask")
        print("=" * 60)
        print(type(e).__name__)
        print(str(e))
        print("=" * 60)


        return jsonify({

            "error":
                str(e),

            "answer":
                "Sorry, something went wrong while processing your question.",

            "similarity":
                0

        }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("          FAQ AI ASSISTANT")
    print("=" * 60)
    print("Backend : http://127.0.0.1:5000")
    print("Health  : http://127.0.0.1:5000/health")
    print("Frontend: http://127.0.0.1:5500")
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )