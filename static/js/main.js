// API Base URL
const API_BASE = window.location.origin;

// DOM Elements
const predictionForm = document.getElementById('prediction-form');
const resultCard = document.getElementById('result-card');
const errorMessage = document.getElementById('error-message');
const btnText = document.getElementById('btn-text');
const btnLoading = document.getElementById('btn-loading');

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadModelInfo();
    loadHistory();
});

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/api/model-info`);
        if (!response.ok) throw new Error('Failed to load model info');

        const data = await response.json();

        document.getElementById('model-name').textContent = data.model_name;
        document.getElementById('model-mae').textContent = `₼${data.metrics.test_mae}`;
        document.getElementById('model-r2').textContent = data.metrics.test_r2;
    } catch (error) {
        console.error('Error loading model info:', error);
        showError('Failed to load model information');
    }
}

// Load historical data
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/history`);
        if (!response.ok) throw new Error('Failed to load history');

        const data = await response.json();

        // Update statistics
        document.getElementById('avg-monthly').textContent = `₼${formatNumber(data.statistics.average_monthly)}`;
        document.getElementById('median-monthly').textContent = `₼${formatNumber(data.statistics.median_monthly)}`;
        document.getElementById('max-monthly').textContent = `₼${formatNumber(data.statistics.max_monthly)}`;
        document.getElementById('min-monthly').textContent = `₼${formatNumber(data.statistics.min_monthly)}`;

        // Update table
        const tbody = document.getElementById('history-tbody');
        tbody.innerHTML = '';

        data.history.reverse().forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${item.year_month}</td>
                <td>₼${formatNumber(item.total_amount)}</td>
                <td>${item.transaction_count}</td>
            `;
            tbody.appendChild(row);
        });
    } catch (error) {
        console.error('Error loading history:', error);
        document.getElementById('history-tbody').innerHTML =
            '<tr><td colspan="3" class="loading">Failed to load historical data</td></tr>';
    }
}

// Handle form submission
predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const month = parseInt(document.getElementById('month').value);
    const year = parseInt(document.getElementById('year').value);

    if (!month || !year) {
        showError('Please select both month and year');
        return;
    }

    // Show loading state
    btnText.classList.add('hidden');
    btnLoading.classList.remove('hidden');
    resultCard.classList.add('hidden');
    hideError();

    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ month, year })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }

        const data = await response.json();
        displayPrediction(data);

    } catch (error) {
        console.error('Error making prediction:', error);
        showError(error.message || 'Failed to make prediction. Please try again.');
    } finally {
        // Reset button state
        btnText.classList.remove('hidden');
        btnLoading.classList.add('hidden');
    }
});

// Display prediction result
function displayPrediction(data) {
    // Update month/year
    document.getElementById('result-month').textContent = `${data.month} ${data.year}`;

    // Update predicted amount
    document.getElementById('predicted-amount').textContent = formatNumber(data.predicted_expense);

    // Update confidence intervals
    document.getElementById('ci68-lower').textContent = formatNumber(data.confidence_interval_68.lower);
    document.getElementById('ci68-upper').textContent = formatNumber(data.confidence_interval_68.upper);
    document.getElementById('ci95-lower').textContent = formatNumber(data.confidence_interval_95.lower);
    document.getElementById('ci95-upper').textContent = formatNumber(data.confidence_interval_95.upper);

    // Show result card with animation
    resultCard.classList.remove('hidden');
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show error message
function showError(message) {
    document.getElementById('error-text').textContent = message;
    errorMessage.classList.remove('hidden');
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Hide error message
function hideError() {
    errorMessage.classList.add('hidden');
}

// Close error message
function closeError() {
    hideError();
}

// Format number with thousand separators
function formatNumber(num) {
    return parseFloat(num).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

// Health check (optional - for debugging)
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Uncomment to run health check on load
// checkHealth();
