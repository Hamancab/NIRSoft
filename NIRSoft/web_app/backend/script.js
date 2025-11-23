document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('analysisForm');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const resultsContainer = document.getElementById('resultsContainer');
  const formContainer = document.getElementById('formContainer');
  const graphList = document.getElementById('graphList');
  const selectedGraph = document.getElementById('selectedGraph');
  const newAnalysisBtn = document.getElementById('newAnalysisBtn');
  const viewPdfBtn = document.getElementById('viewPdfBtn');

  // PDF butonuna tıklanınca PDF'i aç
  viewPdfBtn.onclick = function() {
    if (window.currentPdfPath) {
      window.open(window.currentPdfPath, '_blank');
    }
  };

  // Grafik listesini doldur
  function showResults(result) {
    // PDF yolunu kaydet
    window.currentPdfPath = result.pdf_path;
    // Grafik listesini doldur
    const graphList = document.getElementById('graphList');
    const selectedGraph = document.getElementById('selectedGraph');
    graphList.innerHTML = '';
    if (result.pngs && result.pngs.length > 0) {
      result.pngs.forEach(function(png, idx) {
        const li = document.createElement('li');
        li.textContent = png.split('/').pop();
        li.style.cursor = 'pointer';
        li.onclick = function() {
          selectedGraph.src = png;
          // Seçili grafiği vurgula
          Array.from(graphList.children).forEach(x => x.classList.remove('selected'));
          li.classList.add('selected');
        };
        graphList.appendChild(li);
      });
      // Varsayılan olarak ilk grafik seçili olsun
      selectedGraph.src = result.pngs[0];
      if (graphList.children[0]) graphList.children[0].classList.add('selected');
    } else {
      selectedGraph.src = '';
    }
  }

  // Analiz sonrası sonucu göster
  function handleAnalysisResponse(data) {
    document.getElementById('formContainer').classList.add('hidden');
    document.getElementById('resultsContainer').classList.remove('hidden');
    showResults(data.result);
  }

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
        handleAnalysisResponse(data);
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
