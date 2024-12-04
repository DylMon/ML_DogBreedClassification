// Define the model URL (update this with your model's path)
const MODEL_URL = './model/model.json';

// Load the model
let model;
async function loadModel() {
    document.getElementById("predictionResult").textContent = "Loading model...";
    model = await tf.loadLayersModel(MODEL_URL);
    document.getElementById("predictionResult").textContent = "Model loaded. Ready for predictions!";
}
loadModel();

// Handle file input and preview
document.getElementById("fileInput").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (file) {
        // Display the uploaded image
        const imgPreview = document.getElementById("imagePreview");
        imgPreview.src = URL.createObjectURL(file);

        // Wait for the image to load before predicting
        imgPreview.onload = () => predict(imgPreview);
    }
});

// Predict the breed
async function predict(imageElement) {
    if (!model) {
        alert("Model not loaded yet!");
        return;
    }

    // Preprocess the image
    const tensor = tf.browser.fromPixels(imageElement) // Load image as tensor
        .resizeBilinear([224, 224]) // Resize to model's input size
        .div(255.0) // Normalize to [0, 1]
        .expandDims(0); // Add batch dimension

    // Make a prediction
    const predictions = model.predict(tensor).dataSync(); // Get prediction scores
    const predictedIndex = tf.argMax(predictions).dataSync()[0]; // Get index of highest score
    const confidence = (predictions[predictedIndex] * 100).toFixed(2); // Confidence score

    // List of breeds (replace with your dataset's breed names)
    const breeds = ["Breed 1", "Breed 2", "Breed 3"]; // Update with actual breed names

    // Show the result
    const predictedBreed = breeds[predictedIndex];
    document.getElementById("predictionResult").textContent =
        `Prediction: ${predictedBreed} (Confidence: ${confidence}%)`;
}
