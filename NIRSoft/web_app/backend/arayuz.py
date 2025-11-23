from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import analysis
import shutil
import scipy.io
import numpy as np

# Flask uygulamasına CORS ekle
app = Flask(__name__, static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../static')))
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../static'))
STATIC_RESULTS_FOLDER = UPLOAD_FOLDER  # Artık tüm sonuçlar static'e kaydedilecek
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_RESULTS_FOLDER, exist_ok=True)

@app.route('/')
def serve_index():
    return send_from_directory('../frontend_V3', 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory('../frontend_V3', path)

@app.route('/api/data', methods=['POST'])
def process_data():
    # Form verilerini al
    name = request.form.get('name')
    patient_id = request.form.get('patient_id')
    dob = request.form.get('dob')
    analysis_date = request.form.get('analysis_date')
    gender = request.form.get('gender')
    protocol_name = request.form.get('protocol_name')
    diagnosis = request.form.get('diagnosis')
    notes = request.form.get('notes')
    nirs_file = request.files.get('nirs_file')

    file_path = None
    result = None
    if nirs_file:
        print('--- Analiz başlatıldı ---')
        file_path = os.path.join(UPLOAD_FOLDER, nirs_file.filename)
        nirs_file.save(file_path)
        try:
            probe_mat_path = os.path.join(os.path.dirname(__file__), 'probe.mat')
            save_dir = STATIC_RESULTS_FOLDER
            os.makedirs(save_dir, exist_ok=True)
            
            analysis_args = dict(
                nirs_file_path=file_path,
                hasta_adi=name or '',
                patient_id=patient_id or '',
                date_of_birth=dob or '',
                gender=gender or '',
                protocol_name=protocol_name or '',
                diagnosis=diagnosis or '',
                notes=notes or '',
                save_dir=save_dir,
                probe_mat_path=probe_mat_path,
                analysis_date=analysis_date or ''
            )

            try:
                analysis.full_analysys(**analysis_args)
                png_files = [f for f in os.listdir(save_dir) if f.endswith('.png')]
                
                # PDF'yi static klasörüne taşı
                pdf_source_path = os.path.join(os.path.dirname(__file__), 'rapor.pdf')
                pdf_target_path = os.path.join(STATIC_RESULTS_FOLDER, 'rapor.pdf')
                if os.path.exists(pdf_source_path):
                    shutil.move(pdf_source_path, pdf_target_path)
                else:
                    print(f"PDF bulunamadı: {pdf_source_path}")
                
                return jsonify({
                    'result': {
                        'pngs': png_files,
                        'pdf_path': '/static/rapor.pdf'
                    }
                })
            except Exception as e:
                print(f"--- Analiz HATASI ---\n{str(e)}")
                return jsonify({'error': str(e)}), 500
        except Exception as e:
            print('--- Analiz HATASI ---')
            print(str(e))
            return jsonify({'status': 'error', 'message': str(e)})

    if nirs_file and nirs_file.filename.endswith('.mat'):
        try:
            print(f"Dosya okunuyor: {file_path}")
            mat = scipy.io.loadmat(file_path)
            print(f"Dosya içeriği: {mat.keys()}")
            
            if 't' in mat and 's' in mat:
                t = mat['t'].squeeze()
                s = mat['s']
                print(f"Trigger matrisleri bulundu: t ({t.shape}), s ({s.shape})")
                
                trigger_indices = sorted({
                    idx
                    for ch in range(s.shape[1])
                    for idx in np.where(s[:, ch] == 1)[0]
                })
                trigger_times = [t[idx] for idx in trigger_indices]

                return jsonify({
                    'status': 'success',
                    'triggers': {
                        'indices': trigger_indices,
                        'times': trigger_times
                    }
                })
            else:
                print("Trigger matrisleri bulunamadı.")
                return jsonify({'status': 'error', 'message': 'Trigger matrisleri bulunamadı.'})
        except Exception as e:
            print('--- Analiz HATASI ---')
            print(str(e))
            return jsonify({'status': 'error', 'message': str(e)})

    response = {
        "status": "success",
        "message": "Data and file received successfully.",
        "file_path": file_path,
        "form_data": {
            "name": name,
            "patient_id": patient_id,
            "dob": dob,
            "analysis_date": analysis_date,
            "gender": gender,
            "protocol_name": protocol_name,
            "diagnosis": diagnosis,
            "notes": notes
        },
        "result": result
    }
    return jsonify(response)

@app.route('/api/graphs')
def get_graphs():
    static_dir = app.static_folder
    png_files = [f for f in os.listdir(static_dir) if f.endswith('.png')]
    return jsonify(sorted(png_files))

@app.route('/static/<path:filename>')
def serve_static_file(filename):
    return send_from_directory(STATIC_RESULTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)