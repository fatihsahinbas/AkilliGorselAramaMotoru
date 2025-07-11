import os
import pickle
from PIL import Image
from akilli_gorsel_arama.feature_extractor import FeatureExtractor
from tqdm import tqdm # İlerleme çubuğu için güzel bir kütüphane

# pip install tqdm

def index_images(dataset_path='dataset'):
    extractor = FeatureExtractor()
    features = []
    img_paths = []

    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    print(f"Toplam {len(image_files)} adet görsel indeksleniyor...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(dataset_path, img_name)
        try:
            with Image.open(img_path) as img:
                feature_vector = extractor.extract(img)
                features.append(feature_vector)
                img_paths.append(img_path)
        except Exception as e:
            print(f"Hata: {img_path} işlenemedi. - {e}")
            
    index_data = {
        'paths': img_paths,
        'features': features
    }

    with open('features.pkl', 'wb') as f:
        pickle.dump(index_data, f)
        
    print(f"\nİndeksleme tamamlandı. {len(features)} görselin vektörü 'features.pkl' dosyasına kaydedildi.")

if __name__ == '__main__':
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
        print("'dataset' klasörü oluşturuldu. Lütfen arama yapılacak görselleri bu klasöre ekleyin.")
    else:
        index_images()