// ============================================================
// FAQ AI ASSISTANT
// ============================================================

console.log("FAQ AI Assistant loaded");


// Backend URL

const API_URL =
    "http://127.0.0.1:5000/ask";


// HTML elements

const chatBox =
    document.getElementById("chat-box");

const questionInput =
    document.getElementById("question");

const sendButton =
    document.getElementById("send-button");

const sendText =
    document.getElementById("send-text");


// ============================================================
// ADD MESSAGE
// ============================================================

function addMessage(
    message,
    type,
    similarity = null
) {

    const messageDiv =
        document.createElement("div");

    messageDiv.className =
        `message ${type}-message`;


    // Avatar

    const avatar =
        document.createElement("div");

    avatar.className =
        type === "bot"
            ? "avatar bot-avatar"
            : "avatar user-avatar";

    avatar.textContent =
        type === "bot"
            ? "AI"
            : "YOU";


    // Bubble

    const bubble =
        document.createElement("div");

    bubble.className =
        "bubble";


    // Bot name

    if (type === "bot") {

        const name =
            document.createElement("div");

        name.className =
            "message-name";

        name.textContent =
            "FAQ Assistant";

        bubble.appendChild(name);
    }


    // Message

    const paragraph =
        document.createElement("p");

    paragraph.textContent =
        message;

    bubble.appendChild(paragraph);


    // Similarity score

    if (
        type === "bot" &&
        similarity !== null &&
        similarity > 0
    ) {

        const score =
            document.createElement("div");

        score.style.marginTop =
            "10px";

        score.style.paddingTop =
            "8px";

        score.style.borderTop =
            "1px solid rgba(255,255,255,0.06)";

        score.style.fontSize =
            "10px";

        score.style.color =
            "#64748b";

        score.textContent =
            `Match confidence: ${
                (similarity * 100).toFixed(1)
            }%`;

        bubble.appendChild(score);
    }


    messageDiv.appendChild(
        avatar
    );

    messageDiv.appendChild(
        bubble
    );


    chatBox.appendChild(
        messageDiv
    );


    // Scroll

    chatBox.scrollTop =
        chatBox.scrollHeight;
}


// ============================================================
// SEND QUESTION
// ============================================================

async function sendQuestion() {

    const question =
        questionInput.value.trim();


    if (!question) {

        questionInput.focus();

        return;
    }


    // Show user question

    addMessage(
        question,
        "user"
    );


    // Clear input

    questionInput.value = "";


    // Loading

    sendButton.disabled = true;

    sendText.textContent =
        "Thinking...";


    try {

        console.log(
            "Sending to:",
            API_URL
        );


        // ====================================================
        // SEND REQUEST TO FLASK 5000
        // ====================================================

        const response =
            await fetch(
                API_URL,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            question:
                                question
                        })
                }
            );


        console.log(
            "Backend status:",
            response.status
        );


        // Get JSON

        const data =
            await response.json();


        console.log(
            "Backend response:",
            data
        );


        // Backend error

        if (!response.ok) {

            throw new Error(
                data.error ||
                `Backend error: ${response.status}`
            );
        }


        // ====================================================
        // DISPLAY ANSWER
        // ====================================================

        addMessage(
            data.answer,
            "bot",
            data.similarity
        );


    }

   catch (error) {

        console.error(
            "CHATBOT ERROR:",
            error
        );
    
        addMessage(
            "⚠️ " + error.message,
            "bot"
        );
    }

    finally {

        sendButton.disabled =
            false;

        sendText.textContent =
            "Send";

        questionInput.focus();
    }
}


// ============================================================
// SEND BUTTON
// ============================================================

sendButton.addEventListener(
    "click",
    sendQuestion
);


// ============================================================
// ENTER KEY
// ============================================================

questionInput.addEventListener(
    "keydown",
    function(event) {

        if (
            event.key === "Enter"
        ) {

            event.preventDefault();

            sendQuestion();
        }

    }
);


// ============================================================
// SUGGESTIONS
// ============================================================

function useSuggestion(question) {

    questionInput.value =
        question;

    sendQuestion();
}