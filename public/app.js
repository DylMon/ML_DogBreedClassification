// Define the model URL and breed list paths
const MODEL_URL = './tfjs_model/model.json';
const BREEDS_URL = './breeds.json';

// Initialize variables
let model;
let breeds = [];

// Load the list of breeds
fetch(BREEDS_URL)
    .then(response => response.json())
    .then(data => { breeds = data; })
    .catch(error => console.error("Error loading breed names:", error));

// Load the model
async function loadModel() {
    try {
        document.getElementById("predictionResult").textContent = "Loading model...";
        model = await tf.loadLayersModel(MODEL_URL);
        document.getElementById("predictionResult").textContent = "Model loaded. Ready for predictions!";
    } catch (error) {
        document.getElementById("predictionResult").textContent = "Error loading model!";
        console.error(error);
    }
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
    if (!model || breeds.length === 0) {
        alert("Model or breeds list not loaded yet!");
        return;
    }

    try {
        // Preprocess the image
        const tensor = tf.browser.fromPixels(imageElement)
            .resizeBilinear([224, 224]) // Resize to model's input size
            .div(255.0) // Normalize to [0, 1]
            .expandDims(0); // Add batch dimension

        // Make predictions
        const predictions = model.predict(tensor).dataSync();
        const topK = 3; // Number of top predictions to show
        const sortedIndices = Array.from(predictions)
            .map((p, i) => [p, i]) // Pair scores with indices
            .sort((a, b) => b[0] - a[0]) // Sort by scores descending
            .slice(0, topK); // Get top K predictions

        // Display the top predictions
        let resultText = "Top predictions:\n";
        sortedIndices.forEach(([score, index]) => {
            resultText += `${breeds[index]}: ${(score * 100).toFixed(2)}%\n`;
        });
        document.getElementById("predictionResult").textContent = resultText;

    } catch (error) {
        console.error("Error during prediction:", error);
        document.getElementById("predictionResult").textContent = "Error during prediction.";
    }
}
