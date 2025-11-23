// This file contains the JavaScript code for the frontend application.
// It handles user interactions, makes API calls to the backend, and updates the UI accordingly.

document.addEventListener('DOMContentLoaded', function() {
    const resultDiv = document.getElementById('result');

    document.getElementById('analysis-form').addEventListener('submit', function(e) {
        e.preventDefault();
        sendData();
    });

    function showSpinner() {
        document.getElementById('loading-spinner').style.display = 'flex';
    }

    function hideSpinner() {
        document.getElementById('loading-spinner').style.display = 'none';
    }

    function sendData() {
        const form = document.getElementById('analysis-form');
        const formData = new FormData(form);
        const fileInput = document.getElementById('nirs_file');
        if (!fileInput.files.length) {
            alert('Lütfen bir .nirs dosyası seçin!');
            return;
        }
        showSpinner();
        fetch('/api/data', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideSpinner();
            if (data.status === 'success' && data.result && data.result.pdf_path) {
                const pdfSection = document.getElementById('pdf-section');
                pdfSection.style.display = 'block';
                const openPdfBtn = document.getElementById('open-pdf-btn');
                openPdfBtn.onclick = function() {
                    window.open('/static/' + data.result.pdf_path.split('static/').pop(), '_blank');
                };
            }
            // Grafikler
            if (data.result && data.result.pngs && data.result.pngs.length > 0) {
                const graphsSection = document.getElementById('graphs-section');
                const graphList = document.getElementById('graph-list');
                const graphPreview = document.getElementById('graph-preview');
                graphsSection.style.display = 'block';
                graphList.innerHTML = '';
                document.body.classList.remove('graphic-selected');
                let selectedIndex = 0;
                function selectGraph(idx) {
                    const items = graphList.querySelectorAll('li');
                    items.forEach((li, i) => {
                        if (i === idx) {
                            li.classList.add('selected');
                        } else {
                            li.classList.remove('selected');
                        }
                    });
                    // Yolun başında zaten /static/ varsa tekrar ekleme
                    let imgSrc = data.result.pngs[idx];
                    if (!imgSrc.startsWith('/static/')) {
                        imgSrc = '/static/' + imgSrc;
                    }
                    graphPreview.src = imgSrc;
                    graphPreview.style.display = 'block';
                    document.body.classList.add('graphic-selected');
                }
                data.result.pngs.forEach(function(png, idx) {
                    const li = document.createElement('li');
                    li.textContent = png.split('/').pop();
                    li.style.cursor = 'pointer';
                    li.onclick = function() {
                        selectGraph(idx);
                    };
                    graphList.appendChild(li);
                });
                // Varsayılan olarak ilk grafik seçili olsun
                selectGraph(0);
            }
            document.getElementById('result').textContent = data.message || '';
        })
        .catch(err => {
            hideSpinner();
            document.getElementById('result').textContent = 'Sunucu hatası: ' + err;
        });
    }
});