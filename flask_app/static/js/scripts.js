function showLoader() {
    document.getElementById('loader').style.display = 'block';
}

function hideLoader() {
    document.getElementById('loader').style.display = 'none';
}

function submitForm() {
    var formData = new FormData();
    formData.append('modelType', document.getElementById('modelType').value);
    formData.append('inputType', document.getElementById('inputType').value);

    if (formData.get('inputType') === 'direct') {
        var sequences = document.getElementById('sequences').value.split(',').map(seq => seq.trim()).filter(seq => seq);
        formData.append('sequences', sequences.join(','));
    } else if (formData.get('inputType') === 'csv' || formData.get('inputType') === 'fasta') {
        var file = document.getElementById('file').files[0];
        if (!file) {
            alert('Please select a file');
            return;
        }
        formData.append('file', file);
    }

    sendPredictRequest(formData);
}

function sendPredictRequest(formData) {
    showLoader();  // Show loader before making the request
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        hideLoader();  // Hide loader after getting the response
        if (!response.ok) {
            throw new Error('Request failed.');
        }
        return response.json();
    })
    .then(data => {
        displayResults(data);
    })
    .catch(error => {
        hideLoader();  // Hide loader if there is an error
        console.error('Error:', error);
        displayError('Request failed. Please check your input and try again.');
    });
}

function displayResults(data) {
    var resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = ''; 

    if (data.error) {
        displayError(data.error);
    } else {
        var html = '<h2>Results:</h2>';
        html += '<ul class="results-list">';
        data.results.forEach(result => {
            let truncatedSequence = result.sequence.length > 50 
                ? result.sequence.substring(0, 47) + '...' 
                : result.sequence;
            html += `<li>
                <div class="sequence-wrapper" title="${result.sequence}">
                    ${truncatedSequence}
                </div>
                : ${result.prediction} (Probability: ${result.probability.toFixed(2)})
            </li>`;
        });
        html += '</ul>';
        resultsDiv.innerHTML = html;
    }
}

function displayError(message) {
    var resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `<p class="error">${message}</p>`;
}

document.getElementById('inputType').addEventListener('change', function() {
    var inputType = this.value;
    if (inputType === 'direct') {
        document.getElementById('sequenceInput').style.display = 'block';
        document.getElementById('fileInput').style.display = 'none';
    } else if (inputType === 'csv' || inputType === 'fasta') {
        document.getElementById('sequenceInput').style.display = 'none';
        document.getElementById('fileInput').style.display = 'block';
    }
});

// Menu toggle
document.getElementById('hamburger').addEventListener('click', function() {
    document.getElementById('menu').classList.add('active');
});

document.getElementById('closeBtn').addEventListener('click', function() {
    document.getElementById('menu').classList.remove('active');
});
