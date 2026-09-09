const tweetForm = document.getElementById("tweetForm");
const tweetText = document.getElementById("tweetText");
const characterCounter = document.getElementById("characterCounter");
const resultBox = document.getElementById("resultBox");
const analyzeButton = tweetForm.querySelector("button[type='submit']");
const exampleButtons = document.querySelectorAll(".example-chip");

function updateCounter() {
    const characterCount = tweetText.value.length;
    characterCounter.textContent = `${characterCount} / 280`;
    characterCounter.style.color = characterCount > 260 ? "var(--coral)" : "";
}

function showMessage(message, type = "error") {
    resultBox.className = `result-box ${type}`;
    resultBox.replaceChildren();
    const messageText = document.createElement("p");
    messageText.textContent = message;
    resultBox.appendChild(messageText);
}

function renderResult(data) {
    const sentimentClass = data.sentiment.toLowerCase();
    resultBox.className = `result-box ${sentimentClass}`;
    resultBox.replaceChildren();

    const kicker = document.createElement("p");
    kicker.className = "result-kicker";
    kicker.textContent = "MODEL READING / COMPLETE";

    const heading = document.createElement("h2");
    heading.textContent = data.sentiment;

    const explanationLabel = document.createElement("strong");
    explanationLabel.textContent = "Why it reads this way";

    const explanationText = document.createElement("p");
    explanationText.textContent = data.explanation;
    resultBox.append(kicker, heading, explanationLabel, explanationText);

    if (data.translation_applied) {
        const translationLabel = document.createElement("strong");
        translationLabel.className = "translation";
        translationLabel.textContent = "English translation";
        const translationText = document.createElement("p");
        translationText.textContent = data.translated_tweet;
        resultBox.append(translationLabel, translationText);
    }
}

tweetText.addEventListener("input", updateCounter);

exampleButtons.forEach((button) => {
    button.addEventListener("click", () => {
        tweetText.value = button.dataset.example;
        updateCounter();
        tweetText.focus();
    });
});

tweetForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const tweet = tweetText.value.trim();

    if (!tweet) {
        showMessage("Please enter a tweet.");
        tweetText.focus();
        return;
    }

    if (tweet.length > 280) {
        showMessage("The tweet must not exceed 280 characters.");
        return;
    }

    analyzeButton.disabled = true;
    analyzeButton.querySelector("span").textContent = "Reading...";
    resultBox.className = "result-box";
    resultBox.replaceChildren();
    const loadingText = document.createElement("p");
    loadingText.textContent = "Running the local model...";
    resultBox.appendChild(loadingText);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tweet })
        });
        const contentType = response.headers.get("content-type") || "";
        const data = contentType.includes("application/json")
            ? await response.json()
            : { error: "The server returned an invalid response. Please try again." };
        if (!response.ok) throw new Error(data.error || "Prediction failed.");
        renderResult(data);
    } catch (error) {
        showMessage(error.message);
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.querySelector("span").textContent = "Analyze sentiment";
    }
});

updateCounter();
