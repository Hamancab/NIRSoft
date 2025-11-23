document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('analysisForm');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const resultsContainer = document.getElementById('resultsContainer');
  const formContainer = document.getElementById('formContainer');
  const graphList = document.getElementById('graphList');
  const selectedGraph = document.getElementById('selectedGraph');
  const newAnalysisBtn = document.getElementById('newAnalysisBtn');
  const viewPdfBtn = document.getElementById('viewPdfBtn');

  form.addEventListener('submit', function(e) {
    e.preventDefault();
    const nirsFileInput = document.getElementById('nirs_file');
    if (nirsFileInput.files.length === 0) {
      alert('Lütfen bir .nirs dosyası seçiniz.');
      return;
    }

    const formData = new FormData(form);
    loadingOverlay.classList.remove('hidden');

    fetch('/api/data', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      loadingOverlay.classList.add('hidden');
      if (data.status === 'success') {
        formContainer.classList.add('hidden');
        resultsContainer.classList.remove('hidden');

        const pngs = data.result.pngs || [];
        graphList.innerHTML = '';
        pngs.forEach(function(png) {
          const li = document.createElement('li');
          li.textContent = png;
          li.addEventListener('click', function() {
            selectedGraph.src = `/static/results/${png}`;
          });
          graphList.appendChild(li);
        });

        const pdfName = data.result.pdf_path;
        if (pdfName) {
          viewPdfBtn.style.display = 'inline-block';
          viewPdfBtn.onclick = function() {
            window.open(`/static/results/${pdfName}`, '_blank');
          };
        } else {
          viewPdfBtn.style.display = 'none';
        }
      } else {
        alert('Analiz sırasında hata oluştu: ' + data.message);
      }
    })
    .catch(error => {
      loadingOverlay.classList.add('hidden');
      alert('İstek göndermede hata: ' + error);
    });
  });

  newAnalysisBtn.addEventListener('click', function() {
    form.reset();
    resultsContainer.classList.add('hidden');
    formContainer.classList.remove('hidden');
    selectedGraph.src = '';
  });
});
