const EMOTION_COLORS = {
    sadness: "var(--sadness)", joy: "var(--joy)", love: "var(--love)",
    anger: "var(--anger)", fear: "var(--fear)", surprise: "var(--surprise)",
    neutral: "var(--muted)"
};

let planStatus = null;

const byId = (id) => document.getElementById(id);

const formatDate = (epochSeconds) =>
    new Date(epochSeconds * 1000).toLocaleDateString(undefined,
        { day: "numeric", month: "short", year: "numeric" });

const clearNode = (node) => { while (node.firstChild) { node.removeChild(node.firstChild); } };

const showMessage = (kind, message) => {
    const box = byId("system_response");
    clearNode(box);
    const alertBox = document.createElement("div");
    alertBox.className = "alert alert-" + kind;
    alertBox.setAttribute("role", "alert");
    alertBox.textContent = message;
    box.appendChild(alertBox);
};

const renderResult = (data) => {
    const box = byId("system_response");
    clearNode(box);

    const title = document.createElement("h2");
    title.className = "result-title";
    title.textContent = "Result of Emotion Detection";
    box.appendChild(title);

    const dominant = data.dominant_emotion;
    const verdict = document.createElement("p");
    verdict.className = "verdict";
    verdict.style.color = EMOTION_COLORS[dominant] || "var(--ink)";
    verdict.textContent = dominant === "neutral" ? "No strong emotion" : dominant;
    box.appendChild(verdict);

    const note = document.createElement("p");
    note.className = "verdict-note";
    note.textContent = dominant === "neutral"
        ? "The model is not confident about any one emotion here. Try a sentence about how someone feels."
        : "This is the dominant emotion in the text.";
    box.appendChild(note);

    const list = document.createElement("ul");
    list.className = "bars";
    Object.keys(data.scores).forEach((name) => {
        const score = data.scores[name] || 0;
        const row = document.createElement("li");
        if (name === dominant) { row.className = "is-top"; }

        const label = document.createElement("span");
        label.className = "bar-name";
        label.textContent = name;

        const track = document.createElement("span");
        track.className = "bar-track";
        const fill = document.createElement("span");
        fill.className = "bar-fill";
        fill.style.background = EMOTION_COLORS[name] || "var(--muted)";
        track.appendChild(fill);

        const value = document.createElement("span");
        value.className = "bar-value";
        value.textContent = Math.round(score * 100) + "%";

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(value);
        list.appendChild(row);
        requestAnimationFrame(() => { fill.style.width = Math.round(score * 100) + "%"; });
    });
    box.appendChild(list);
};

const updateCharCount = () => {
    const counter = byId("char-count");
    const length = byId("textToAnalyze").value.length;
    if (!planStatus) { counter.textContent = ""; return; }
    counter.textContent = length.toLocaleString() + " / " + planStatus.max_chars.toLocaleString();
    counter.classList.toggle("warn", length > planStatus.max_chars);
};

const renderStatus = (status) => {
    planStatus = status;
    const usage = byId("usage");
    const upgrade = byId("upgrade");
    const proNote = byId("pro-note");

    if (status.tier === "demo") {
        usage.textContent = "";
        upgrade.classList.add("hidden");
        proNote.classList.add("hidden");
    } else if (status.tier === "pro") {
        usage.textContent = "Pro: unlimited analyses";
        upgrade.classList.add("hidden");
        proNote.classList.remove("hidden");
        byId("pro-copy").textContent = "Unlimited analyses and texts up to " +
            status.max_chars.toLocaleString() + " characters, until " + formatDate(status.expires_at) + ".";
    } else {
        usage.textContent = status.remaining + " of " + status.daily_limit + " free analyses left today";
        usage.classList.toggle("warn", status.remaining === 0);
        proNote.classList.add("hidden");
        upgrade.classList.remove("hidden");
        byId("upgrade-copy").textContent = "\u20B9" + status.price_inr + " for " + status.pro_days +
            " days. Unlimited analyses and texts up to " + status.pro_max_chars.toLocaleString() + " characters.";
        const buy = byId("buy-btn");
        buy.disabled = !status.payments_enabled;
        if (!status.payments_enabled) {
            byId("upgrade-msg").textContent = "Payments are not enabled on this server yet.";
        }
    }
    updateCharCount();
};

const loadStatus = async () => {
    try {
        const response = await fetch("api/status");
        renderStatus(await response.json());
    } catch (error) {
        byId("usage").textContent = "";
    }
};

const loadModelInfo = async () => {
    try {
        const response = await fetch("api/model");
        if (!response.ok) { return; }
        const info = await response.json();
        if (!info.train_docs || info.test_accuracy === undefined) { return; }
        const note = byId("model-note");
        note.textContent = "Trained from scratch on " + info.train_docs.toLocaleString() +
            " texts. Accuracy on " + info.test_docs.toLocaleString() + " texts it never saw: " +
            Math.round(info.test_accuracy * 100) + "%.";
        note.classList.remove("hidden");
    } catch (error) {
        return;
    }
};

let RunSentimentAnalysis = async () => {
    const input = byId("textToAnalyze");
    const textToAnalyze = input.value.trim();
    const button = byId("run-btn");

    if (!textToAnalyze) {
        showMessage("warning", "Enter some text to analyze.");
        input.focus();
        return;
    }

    button.disabled = true;
    const originalLabel = button.textContent;
    button.textContent = "Analyzing...";
    try {
        const response = await fetch("emotionDetector?format=json", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ textToAnalyze: textToAnalyze })
        });
        const data = await response.json();
        if (!response.ok) {
            showMessage(response.status === 402 || response.status === 413 ? "warning" : "danger",
                data.error || "Something went wrong. Please try again.");
            if (data.upgrade) { byId("upgrade").scrollIntoView({ behavior: "smooth", block: "center" }); }
        } else {
            renderResult(data);
        }
        if (data.status) { renderStatus(Object.assign({}, planStatus, data.status)); }
        else { loadStatus(); }
    } catch (error) {
        showMessage("danger", "Could not reach the server. Check your connection and try again.");
    } finally {
        button.disabled = false;
        button.textContent = originalLabel;
    }
};

const loadRazorpay = () => new Promise((resolve, reject) => {
    if (window.Razorpay) { resolve(); return; }
    const script = document.createElement("script");
    script.src = "https://checkout.razorpay.com/v1/checkout.js";
    script.onload = resolve;
    script.onerror = () => reject(new Error("Could not load the payment window."));
    document.head.appendChild(script);
});

const postJson = async (url, payload) => {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {})
    });
    const data = await response.json();
    if (!response.ok) { throw new Error(data.error || "Something went wrong."); }
    return data;
};

const setUpgradeMessage = (message) => { byId("upgrade-msg").textContent = message; };

const startCheckout = async () => {
    const buy = byId("buy-btn");
    buy.disabled = true;
    setUpgradeMessage("Opening the payment window...");
    try {
        const [order] = await Promise.all([postJson("api/create-order"), loadRazorpay()]);
        const checkout = new window.Razorpay({
            key: order.key_id,
            amount: order.amount,
            currency: order.currency,
            name: order.name,
            description: order.description,
            order_id: order.order_id,
            theme: { color: "#172033" },
            modal: { ondismiss: () => { buy.disabled = false; setUpgradeMessage(""); } },
            handler: async (payment) => {
                setUpgradeMessage("Confirming your payment...");
                try {
                    const result = await postJson("api/verify-payment", payment);
                    byId("license-key-text").textContent = result.license_key;
                    byId("license-reveal").classList.remove("hidden");
                    await loadStatus();
                } catch (error) {
                    buy.disabled = false;
                    setUpgradeMessage(error.message + " If you were charged, contact support with payment id " +
                        payment.razorpay_payment_id + ".");
                }
            }
        });
        checkout.open();
    } catch (error) {
        buy.disabled = false;
        setUpgradeMessage(error.message);
    }
};

const activateLicense = async () => {
    const key = byId("license-input").value.trim();
    if (!key) { setUpgradeMessage("Enter your license key first."); return; }
    try {
        await postJson("api/activate", { license_key: key });
        await loadStatus();
    } catch (error) {
        setUpgradeMessage(error.message);
    }
};

document.addEventListener("DOMContentLoaded", () => {
    const input = byId("textToAnalyze");
    input.addEventListener("input", updateCharCount);
    input.addEventListener("keydown", (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { RunSentimentAnalysis(); }
    });
    document.querySelectorAll(".chip").forEach((chip) => {
        chip.addEventListener("click", () => {
            input.value = chip.dataset.example;
            updateCharCount();
            input.focus();
        });
    });
    byId("buy-btn").addEventListener("click", startCheckout);
    byId("license-toggle").addEventListener("click", () => {
        byId("license-form").classList.toggle("hidden");
        byId("license-input").focus();
    });
    byId("license-btn").addEventListener("click", activateLicense);
    loadStatus();
    loadModelInfo();
});
