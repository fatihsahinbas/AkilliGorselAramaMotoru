import os
import pickle
from PIL import Image
from tqdm import tqdm
from akilli_gorsel_arama.feature_extractor import FeatureExtractor


def index_images(dataset_path='assets/dataset'):
    extractor = FeatureExtractor()
    features = []
    img_paths = []

    try:
        image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        if not image_files:
            print(f"Uyarı: '{dataset_path}' klasöründe indekslenecek görsel bulunamadı.")
            return
    except FileNotFoundError:
        print(f"Hata: '{dataset_path}' klasörü bulunamadı. Lütfen klasör yapısını kontrol edin.")
        return
    
    print(f"'{dataset_path}' klasöründe toplam {len(image_files)} adet görsel indeksleniyor...")

    for img_name in tqdm(image_files):
        # Gerçek dosya yolunu okumak için kullan
        full_file_path = os.path.join(dataset_path, img_name)
        
        # Webde gösterilecek yolu oluştur (örn: "dataset/1.jpg")
        web_path = os.path.join(os.path.basename(dataset_path), img_name).replace("\\", "/")
        
        try:
            with Image.open(full_file_path) as img:
                feature_vector = extractor.extract(img)
                features.append(feature_vector)
                # Kaydedilecek yol, webde görünecek olan yol olmalı
                img_paths.append(web_path)
        except Exception as e:
            print(f"Hata: {full_file_path} işlenemedi. - {e}")
            
    index_data = {
        'paths': img_paths,
        'features': features
    }

    with open('features.pkl', 'wb') as f:
        pickle.dump(index_data, f)
        
    print(f"\nİndeksleme tamamlandı. {len(features)} görselin vektörü 'features.pkl' dosyasına kaydedildi.")


if __name__ == '__main__':
    TARGET_DATASET_PATH = 'assets/dataset'
    
    if not os.path.exists(TARGET_DATASET_PATH):
        os.makedirs(TARGET_DATASET_PATH)
        print(f"'{TARGET_DATASET_PATH}' klasörü oluşturuldu. Lütfen arama yapılacak görselleri bu klasöre taşıyın.")
    else:
        index_images(dataset_path=TARGET_DATASET_PATH)