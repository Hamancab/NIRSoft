document.addEventListener('DOMContentLoaded', function() {
  const getStartedBtn = document.querySelector('.get-started');
  if (getStartedBtn) {
    getStartedBtn.addEventListener('click', function() {
      window.location.href = 'form.html';
    });
  }

  // Analiz formu için event listener
  if (document.getElementById('analysisForm')) {
    const form = document.getElementById('analysisForm');
    const submitButton = form.querySelector('button[type="submit"]');
    const viewResultButton = form.querySelector('button[type="button"]');
    
    form.addEventListener('submit', async function(e) {
      e.preventDefault();
      
      // Disable buttons
      submitButton.disabled = true;
      viewResultButton.disabled = true;
      
      // Show loading overlay
      const loadingDiv = document.createElement('div');
      loadingDiv.className = 'loading-overlay';
      loadingDiv.innerHTML = `
        <div class="loading-spinner"></div>
        <div class="loading-text">Analiz yapılıyor, lütfen bekleyin...</div>
      `;
      form.appendChild(loadingDiv);
      
      const formData = new FormData(this);
      
      try {
        const response = await fetch('/api/data', { method: 'POST', body: formData });
        const data = await response.json();
        
        if (data.result && data.result.pngs) {
          // Enable buttons
          submitButton.disabled = false;
          viewResultButton.disabled = false;
          
          // Remove loading overlay
          loadingDiv.remove();
          
          // Redirect to results page
          window.location.href = 'sonuc.html?pngs=' + encodeURIComponent(JSON.stringify(data.result.pngs));
        }
      } catch (err) {
        alert('Analiz sırasında bir hata oluştu: ' + err.message);
        submitButton.disabled = false;
        viewResultButton.disabled = false;
        loadingDiv.remove();
      }
    });
  }

  function showLoading() {
    let loadingDiv = document.getElementById('loadingOverlay');
    const targetDiv = document.querySelector('#analysisForm button[type="submit"]').parentElement;
    
    if (!loadingDiv) {
      loadingDiv = document.createElement('div');
      loadingDiv.id = 'loadingOverlay';
      loadingDiv.innerHTML = `
        <div class="loading-spinner" style="width:40px; height:40px; border:3px solid #fff; border-top-color:#ff4b5c; border-radius:50%; animation:spin 1s linear infinite; margin:24px auto;"></div>
        <div style="text-align:center; color:#fff;">Analiz yapılıyor, lütfen bekleyin...</div>
      `;
      targetDiv.appendChild(loadingDiv);
    }
    loadingDiv.style.display = 'block';
  }

  function hideLoading() {
    const loadingDiv = document.getElementById('loadingOverlay');
    if (loadingDiv) {
      loadingDiv.style.display = 'none';
    }
  }

  function showResultButton(params) {
    let resultBtn = document.getElementById('resultButton');
    const targetDiv = document.querySelector('#analysisForm button[type="submit"]').parentElement;
    
    if (!resultBtn) {
      resultBtn = document.createElement('button');
      resultBtn.id = 'resultButton';
      resultBtn.textContent = 'Analiz Sonuçlarını Görüntüle';
      resultBtn.style = `
        display: block;
        margin: 24px auto;
        padding: 16px 32px;
        background: #ff4b5c;
        color: #fff;
        border: none;
        border-radius: 24px;
        font-size: 1.1em;
        font-weight: bold;
        cursor: pointer;
        transition: background 0.2s;
      `;
      targetDiv.appendChild(resultBtn);
    }
    
    resultBtn.onclick = () => window.location.href = 'sonuc.html?' + params.toString();
    resultBtn.style.display = 'block';
  }

  function updateGraphList() {
    const ul = document.getElementById('graphList');
    if (!ul) return;

    fetch('/api/graphs')
      .then(response => response.json())
      .then(data => {
        ul.innerHTML = '';
        data.forEach((graph, index) => {
          const li = document.createElement('li');
          li.innerHTML = `<a href="#" onclick="showGraph('${graph}')">${graph.replace('.png', '').replace(/_/g, ' ')}</a>`;
          ul.appendChild(li);
        });
        
        // İlk grafiği göster
        if (data.length > 0) {
          showGraph(data[0]);
        }
      });
  }

  if (window.location.pathname.includes('sonuc.html')) {
    window.onload = function() {
      updateGraphList();
    };
  }

  window.showGraph = function(graphName) {
    const img = document.getElementById('selectedGraph');
    if (img) {
      img.src = `/static/${graphName}`;
    }
  }

  // PDF açma butonu için event listener
  const pdfButton = document.getElementById('viewPdfBtn');
  if (pdfButton) {
    pdfButton.addEventListener('click', function() {
      window.open('/static/rapor.pdf', '_blank'); // PDF'yi yeni sekmede aç
    });
  }

  const nirsFileInput = document.getElementById('nirs_file');
  const startTriggerSelect = document.getElementById('startTrigger');
  const endTriggerSelect = document.getElementById('endTrigger');
  const submitButton = document.querySelector('button[type="submit"]');

  nirsFileInput.addEventListener('change', async function() {
    const file = nirsFileInput.files[0];
    if (file) {
      const formData = new FormData();
      formData.append('nirs_file', file);

      try {
        const response = await fetch('/api/data', {
          method: 'POST',
          body: formData
        });
        const data = await response.json();

        if (data.status === 'success') {
          const triggers = data.triggers.indices;
          startTriggerSelect.innerHTML = '';
          endTriggerSelect.innerHTML = '';

          triggers.forEach((trigger, index) => {
            const option = document.createElement('option');
            option.value = trigger;
            option.textContent = `Trigger ${index + 1}`;
            startTriggerSelect.appendChild(option);
            endTriggerSelect.appendChild(option.cloneNode(true));
          });

          submitButton.disabled = false; // Triggerlar yüklendikten sonra butonu aktif et
        } else {
          alert('Trigger bilgileri alınamadı: ' + data.message);
          submitButton.disabled = true;
        }
      } catch (error) {
        console.error('Trigger bilgileri alınamadı:', error);
        alert('Trigger bilgileri alınamadı. Lütfen dosyayı kontrol edin.');
        submitButton.disabled = true;
      }
    }
  });
});