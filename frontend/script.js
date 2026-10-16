const form = document.querySelector('form')
const submitBtn = document.querySelector("#submit-btn");
const imagePicker = document.querySelector("#image-picker");
const captionContainer = document.querySelector("#caption-container");
const eventStatus = document.querySelector("#event-status");
const imagePreview = document.querySelector("#image-preview");

imagePicker.addEventListener("change", (event) => {
    const file = event.target.files[0];

    if (file) {

        imagePreview.src = URL.createObjectURL(file);
        imagePreview.style.display = "block";
        

        captionContainer.textContent = "";
        eventStatus.textContent = "";
    } else {
        
        imagePreview.src = "";
        imagePreview.style.display = "none";
    }
});


form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = imagePicker.files[0];
    if (!file) {
        eventStatus.textContent = "Please select an image first.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        submitBtn.disabled = true;
        eventStatus.textContent = "Generating caption, please wait...";
        captionContainer.textContent = "";

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || data.error || "Prediction failed.");
        }

        captionContainer.textContent = data.caption;
        eventStatus.textContent = "Caption generated successfully!";
    } catch (error) {
        console.error("Prediction error:", error);
        eventStatus.textContent = `Error: ${error.message}`;
    } finally {
        submitBtn.disabled = false;
    }
});